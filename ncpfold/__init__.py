"""Per-fold replication of the Phase-1 encoder chain.

Every representation is fitted inside each of the 10 cross-validation folds of
the Giles virtual-trial replica (train side only), then used frozen to encode
both sides of that fold. This removes the one documented deviation of the
Phase-1 chain from the published protocol: encoders trained once on the whole
cohort before the split. The trainers, the experiment registry and the replica
scorer are imported from the main ``neurocausalpfn`` package; nothing is
re-implemented here.
"""
__version__ = "0.1.0"
