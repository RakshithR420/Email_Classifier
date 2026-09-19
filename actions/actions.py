import os

import yaml

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
)

_actions_config = None


def get_actions_config():
    global _actions_config
    if _actions_config is None:
        path = os.path.join(_CONFIG_DIR, "actions.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _actions_config = yaml.safe_load(f)
    return _actions_config


def decide_and_act(service, email, category, dry_run=True):
    """Looks up the configured action for `category` and either logs what
    it would do (dry_run=True) or actually calls the Gmail API (dry_run=False)."""
    from auth.gmail_client import apply_label, archive_message

    config = get_actions_config().get(category, {"action": "none"})
    action = config.get("action", "none")
    label = config.get("label")
    subject = email.get("subject", "(no subject)")

    if action == "none":
        print(f"[{category}] no action configured — {subject}")
        return

    if dry_run:
        print(f"[DRY RUN] would {action} (label='{label}') -> {subject}")
        return

    if action == "label" and label:
        apply_label(service, email["id"], label)
        print(f"[LIVE] applied label '{label}' -> {subject}")
    elif action == "archive":
        if label:
            apply_label(service, email["id"], label)
        archive_message(service, email["id"])
        print(f"[LIVE] archived (label='{label}') -> {subject}")
    else:
        print(f"[{category}] unknown/misconfigured action '{action}' — {subject}")
