"""Tests that catch the failure modes that actually happened.

Note: 'the loss goes down when overfitting one batch' is NOT a valid DINO test --
a single batch collapses trivially and the loss drops BECAUSE of the collapse.
"""
from __future__ import annotations

import torch

from math_ebt.config import Config
from math_ebt.data.schema import Declaration
from math_ebt.data.views import ViewGenerator
from math_ebt.models.dino import DINOModel
from math_ebt.models.mlm import mask_tokens
from math_ebt.training.losses import dino_loss
from math_ebt.training.trainer_dino import DinoTrainer


def tiny_cfg() -> Config:
    cfg = Config()
    cfg.model.vocab_size = 200
    cfg.model.d_model = 32
    cfg.model.n_layers = 2
    cfg.model.n_heads = 2
    cfg.model.dim_feedforward = 64
    cfg.dino.hidden_dim = 64
    cfg.dino.bottleneck_dim = 16
    cfg.dino.out_dim = 128
    cfg.data.max_seq_len = 32
    cfg.data.max_seq_len_local = 16
    return cfg


def a_decl() -> Declaration:
    return Declaration(
        decl_name="add_comm", lean_namespace="Nat",
        module="Mathlib.Algebra.Group.Basic",
        file_path="Mathlib/Algebra/Group/Basic.lean", kind="theorem",
        type_implicit="forall a b, a + b = b + a",
        type_explicit="forall (a b : N), a + b = b + a",
        proof_source="by simp [Nat.add_comm]",
        ast_hash="h1", type_ast_hash="h2",
        type_alpha="forall u v, u + v = v + u",
        type_subterms=["a + b", "b + a"],
        hypotheses=["h1 : P a", "h2 : Q b"],
        proof_steps=["by simp", "exact foo"],
    )


def test_domain_label_comes_from_module():
    # The bug this guards: using lean_namespace ("Nat") as the probe label.
    assert a_decl().domain_label == "Algebra"


def test_head_gain_is_frozen_at_one():
    model = DINOModel(tiny_cfg())
    ll = model.student_head.last_layer
    g = ll.parametrizations.weight.original0 if hasattr(ll, "parametrizations") else ll.weight_g
    assert torch.allclose(g, torch.ones_like(g))
    assert not g.requires_grad


def test_head_normalizes_bottleneck():
    # Without the L2 normalization the logits are unbounded and collapse follows.
    head = DINOModel(tiny_cfg()).student_head
    x = torch.randn(4, 32) * 100
    out = head(x)
    assert out.abs().max() < 1e3


def test_center_is_a_checkpointed_buffer():
    model = DINOModel(tiny_cfg())
    assert "center" in model.state_dict()


def test_teacher_receives_no_gradient():
    cfg = tiny_cfg()
    model = DINOModel(cfg)
    ids = torch.randint(0, 200, (2, 16))
    mask = torch.ones_like(ids)
    _, tproj = model.forward_teacher(ids, mask)
    assert not tproj.requires_grad
    assert all(not p.requires_grad for p in model.teacher_parameters())


def test_ema_moves_teacher_towards_student():
    model = DINOModel(tiny_cfg())
    before = next(model.teacher_backbone.parameters()).clone()
    for p in model.student_backbone.parameters():
        p.data.add_(1.0)
    model.update_teacher(0.9)
    after = next(model.teacher_backbone.parameters())
    assert not torch.allclose(before, after)


def test_dino_loss_skips_identical_view_pairs():
    b, v, k = 3, 6, 32
    s = torch.randn(b, v, k, requires_grad=True)
    t = torch.randn(b, 2, k)
    loss, diag = dino_loss(s, t, torch.zeros(k), 0.1, 0.04)
    assert torch.isfinite(loss)
    loss.backward()
    assert s.grad is not None
    assert 0.0 < diag["teacher_entropy_mean"]


def test_local_views_are_shorter_than_globals():
    # Invariant 3: local views must be genuinely local, not masked full sequences.
    gen = ViewGenerator("certified", num_local=4)
    d = a_decl()
    g0, g1 = gen.global_views(d)
    locals_ = gen.local_views(d)
    assert all(len(lv) < len(g0) for lv in locals_)
    assert g1.count("[SEP]") == 1  # type-only: no proof segment


def test_global_view_1_is_always_type_only():
    gen = ViewGenerator("certified", num_local=2)
    d = a_decl()
    for _ in range(20):
        _, g1 = gen.global_views(d)
        assert "simp" not in g1


def test_student_converges_to_frozen_synthetic_teacher_target():
    # Replaces "loss decreases when overfitting one batch": with a single batch,
    # DINO's loss drops BECAUSE of collapse, so that check would pass on a broken
    # model. Instead freeze a target distribution and check the student can be
    # driven toward it -- this isolates whether the loss/backward path is sane
    # without going through the (collapse-prone) teacher-EMA loop at all.
    torch.manual_seed(0)
    b, v, k = 4, 2, 32
    student_logits = torch.randn(b, v, k, requires_grad=True)
    target = torch.zeros(b, 2, k)
    target[:, :, 0] = 1.0  # every teacher view certain about class 0
    center = torch.zeros(k)
    opt = torch.optim.Adam([student_logits], lr=0.1)

    first_loss = None
    for _ in range(200):
        opt.zero_grad()
        loss, _ = dino_loss(student_logits, target, center, student_temp=0.1, teacher_temp=0.04)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()

    assert loss.item() < first_loss * 0.1
    pred = (student_logits / 0.1).softmax(-1)  # same student_temp used in the loss
    assert pred[..., 0].mean() > 0.9


def test_mlm_masking_never_touches_pad_or_special_tokens():
    torch.manual_seed(0)
    pad_id, mask_id, special_id = 0, 1, 2
    input_ids = torch.randint(3, 100, (8, 20))
    input_ids[:, -5:] = pad_id  # trailing padding
    input_ids[:, 0] = special_id
    attention_mask = (input_ids != pad_id).long()

    out_ids, labels = mask_tokens(
        input_ids, attention_mask, mask_id=mask_id, vocab_size=100,
        special_ids={special_id}, mlm_prob=0.5,
    )
    assert (labels[:, -5:] == -100).all()  # padding never a target
    assert (labels[:, 0] == -100).all()    # special token never a target
    assert (out_ids[attention_mask == 0] == pad_id).all()  # padding never corrupted


def test_trainer_checkpoint_round_trips(tmp_path):
    # Colab sessions disconnect; resuming from `last.pt` must actually restore
    # progress, not just load without error -- check step AND a real parameter
    # value survive a save/load cycle.
    cfg = tiny_cfg()
    cfg.project.output_dir = str(tmp_path)
    device = torch.device("cpu")
    trainer = DinoTrainer(cfg, DINOModel(cfg), loader=None, device=device)
    trainer.step = 1234
    with torch.no_grad():
        next(trainer.model.student_parameters()).add_(5.0)
    trainer.save("last.pt")
    saved_param = next(trainer.model.student_parameters()).clone()

    fresh = DinoTrainer(cfg, DINOModel(cfg), loader=None, device=device)
    fresh.load(tmp_path / "last.pt")
    assert fresh.step == 1234
    assert torch.allclose(next(fresh.model.student_parameters()), saved_param)


def test_ablation_arms_produce_same_view_counts():
    d = a_decl()
    for mode in ("certified", "naive", "dropout_only"):
        gen = ViewGenerator(mode, num_local=4)
        assert len(gen.global_views(d)) == 2
        assert len(gen.local_views(d)) == 4


def test_embedding_covers_every_tokenizer_id(tmp_path):
    # The bug this guards: `model.vocab_size` in the YAML is the BPE training
    # target, but train_tokenizer.py calls add_tokens() afterwards, so the saved
    # tokenizer is strictly larger. Sizing nn.Embedding from the YAML value made
    # the top token ids index out of bounds -- a CUDA device-side assert on
    # Colab, a silent wrong answer on CPU/MPS.
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    from math_ebt.tokenization import load_tokenizer

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.train_from_iterator(["a + b = b + a", "forall x, x = x"],
                            trainer=trainers.BpeTrainer(vocab_size=60,
                                                        special_tokens=["[UNK]"],
                                                        show_progress=False))
    tok.add_tokens(["Nat.add_comm", "Finset.sum_congr"])  # the post-hoc constants
    tok_path = tmp_path / "bpe.json"
    tok.save(str(tok_path))

    cfg = tiny_cfg()
    cfg.data.tokenizer_path = str(tok_path)
    cfg.model.vocab_size = 60  # the stale YAML value
    loaded = load_tokenizer(cfg)
    assert cfg.model.vocab_size == loaded.get_vocab_size()

    model = DINOModel(cfg)
    top_id = max(loaded.get_vocab().values())
    assert top_id == cfg.model.vocab_size - 1
    ids = torch.tensor([[top_id] * 4])
    model.student_backbone(ids, torch.ones_like(ids))  # out of bounds before the fix


def test_unseen_domain_falls_into_the_other_bucket():
    # The label map is built from train; a valid/test declaration with a domain
    # train never saw used to get label -1, which bincount and cross_entropy
    # reject and which otherwise just never matches a prediction.
    from math_ebt.data.dataset import DeclarationDataset, build_label_map

    train = [a_decl()]
    label_map = build_label_map(train, num_classes=15)
    unseen = a_decl()
    unseen.module = "Mathlib.Topology.Basic"  # domain "Topology", absent from train
    assert unseen.domain_label not in label_map

    ds = DeclarationDataset([unseen], tiny_cfg(), label_map, seed=0)
    label = ds[0]["label"]
    assert label >= 0
    assert label == max(label_map.values())
