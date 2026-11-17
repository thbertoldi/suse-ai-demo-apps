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


def test_agent_messages_include_deterministic_lifecycle_scenarios():
    agent_messages = get_agent_messages()
    assert any(message.startswith("[demo:lifecycle]") for message in agent_messages)
    assert any(message.startswith("[demo:list-models]") for message in agent_messages)
    assert any(message.startswith("[demo:predict]") for message in agent_messages)


def test_non_kubeflow_mode_only_uses_available_tools():
    # Execute the real selection function with its real prompt list, without
    # starting the generator or an OTLP connection.
    path = os.path.join(os.path.dirname(__file__), "..", "main.py")
    tree = ast.parse(open(path).read())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "agent_messages")
    namespace = {"AGENT_MESSAGES": get_agent_messages()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    basic = namespace["agent_messages"](False)
    assert basic
    assert all(term not in " ".join(basic).lower() for term in ("iris", "registry", "predict", "[demo:"))
    assert namespace["agent_messages"](True) == get_agent_messages()
