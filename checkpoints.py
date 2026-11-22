"""Portable checkpoint loading for teacher and student policies."""
import torch

from teacher_model import TeacherActorCritic
from student_model import StudentActor


def load_policy(checkpoint_path):
    """Load a privileged teacher or distilled local actor without changing its weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    kind = checkpoint.get("policy_type", "teacher")
    if kind == "teacher":
        if (checkpoint["state_size"], checkpoint["action_size"], checkpoint["observation_order"]) != (
            48, 2, "self_then_other"
        ):
            raise ValueError("Expected a privileged Tennis teacher checkpoint.")
        model = TeacherActorCritic(48, 2, hidden_size=checkpoint["hidden_size"])
    elif kind == "student":
        if (checkpoint["state_size"], checkpoint["action_size"]) != (24, 2):
            raise ValueError("Expected a local Tennis student checkpoint.")
        model = StudentActor(checkpoint["hidden_size"])
    else:
        raise ValueError("Unknown policy type in checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["config"], kind



def load_teacher(checkpoint_path):
    model, config, kind = load_policy(checkpoint_path)
    if kind != "teacher":
        raise ValueError("Demonstrations require a privileged teacher checkpoint.")
    return model, config

