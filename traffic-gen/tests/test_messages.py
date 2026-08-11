import ast
import os


def get_agent_messages():
    """Extract AGENT_MESSAGES from main.py without importing heavy dependencies."""
    main_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
    with open(main_path, "r") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "AGENT_MESSAGES":
                    # Extract list literal
                    if isinstance(node.value, ast.List):
                        return [
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        ]
    return []


def test_agent_messages_include_iris_prediction_prompt():
    agent_messages = get_agent_messages()
    joined = " ".join(agent_messages).lower()
    assert "iris" in joined
    assert "sepal" in joined or "petal" in joined


def test_agent_messages_include_registry_prompt():
    agent_messages = get_agent_messages()
    joined = " ".join(agent_messages).lower()
    assert "model registry" in joined or "registered" in joined
