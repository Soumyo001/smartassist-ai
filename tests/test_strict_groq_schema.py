from task_agent.agent import StrictActionPlan


def test_strict_groq_schema():

    schema = StrictActionPlan.model_json_schema()

    objects = [schema]

    objects.extend(
        definition
        for definition in schema.get(
            "$defs", {}
        ).values()
        if definition.get("type") == "object"
    )

    for obj in objects:
        assert obj["additionalProperties"] is False

        assert set(obj["required"]) == set(
            obj["properties"]
        )