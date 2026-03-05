"""_schema_to_model 及其辅助函数的单元测试。

覆盖：基础类型、嵌套对象、enum、oneOf/anyOf、$ref、description 透传、
安全校验（字段数上限、非法字段名、递归深度上限）等。
"""

from __future__ import annotations

from typing import Any, Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel

from llm_parse.api.routes import (
    _build_object_model,
    _resolve_ref,
    _resolve_type,
    _schema_to_model,
    _to_model_name,
    _validate_field_names,
)


# ---------------------------------------------------------------------------
# _to_model_name
# ---------------------------------------------------------------------------
class TestToModelName:
    def test_snake_case(self) -> None:
        assert _to_model_name("home_address") == "HomeAddress"

    def test_single_word(self) -> None:
        assert _to_model_name("name") == "Name"

    def test_kebab_case(self) -> None:
        assert _to_model_name("my-type") == "MyType"

    def test_empty_string(self) -> None:
        assert _to_model_name("") == "Model"

    def test_mixed_separators(self) -> None:
        assert _to_model_name("a_b-c d") == "ABCD"


# ---------------------------------------------------------------------------
# _resolve_ref
# ---------------------------------------------------------------------------
class TestResolveRef:
    def test_resolve_defs(self) -> None:
        schema: dict[str, Any] = {
            "$defs": {"Address": {"type": "object", "properties": {"city": {"type": "string"}}}},
        }
        result = _resolve_ref("#/$defs/Address", schema)
        assert result["type"] == "object"

    def test_resolve_definitions(self) -> None:
        schema: dict[str, Any] = {
            "definitions": {"Foo": {"type": "string"}},
        }
        result = _resolve_ref("#/definitions/Foo", schema)
        assert result == {"type": "string"}

    def test_external_ref_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _resolve_ref("https://example.com/schema.json", {})
        assert exc_info.value.status_code == 422
        assert "仅支持内部引用" in str(exc_info.value.detail)

    def test_missing_path_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _resolve_ref("#/$defs/Missing", {"$defs": {}})
        assert exc_info.value.status_code == 422
        assert "路径不存在" in str(exc_info.value.detail)

    def test_non_dict_target_raises(self) -> None:
        from fastapi import HTTPException

        schema: dict[str, Any] = {"$defs": {"Bad": "not_a_dict"}}
        with pytest.raises(HTTPException) as exc_info:
            _resolve_ref("#/$defs/Bad", schema)
        assert "不是有效的 Schema 对象" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# _validate_field_names
# ---------------------------------------------------------------------------
class TestValidateFieldNames:
    def test_valid_names(self) -> None:
        _validate_field_names({"name": {}, "age_1": {}, "_private": {}})

    def test_double_underscore_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            _validate_field_names({"__bad": {}})

    def test_invalid_name_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            _validate_field_names({"1abc": {}})


# ---------------------------------------------------------------------------
# _resolve_type — 基础类型
# ---------------------------------------------------------------------------
class TestResolveTypePrimitive:
    @pytest.mark.parametrize(
        ("schema_type", "expected"),
        [("string", str), ("integer", int), ("number", float), ("boolean", bool)],
    )
    def test_primitive_types(self, schema_type: str, expected: type) -> None:
        result = _resolve_type({"type": schema_type}, {}, "f", 0)
        assert result is expected

    def test_default_type_is_string(self) -> None:
        result = _resolve_type({}, {}, "f", 0)
        assert result is str

    def test_unknown_type_falls_back_to_any(self) -> None:
        result = _resolve_type({"type": "unknown_xyz"}, {}, "f", 0)
        assert result is Any


# ---------------------------------------------------------------------------
# _resolve_type — array
# ---------------------------------------------------------------------------
class TestResolveTypeArray:
    def test_array_of_string(self) -> None:
        t = _resolve_type({"type": "array", "items": {"type": "string"}}, {}, "f", 0)
        assert get_origin(t) is list
        assert get_args(t) == (str,)

    def test_array_of_int(self) -> None:
        t = _resolve_type({"type": "array", "items": {"type": "integer"}}, {}, "f", 0)
        assert get_args(t) == (int,)

    def test_array_no_items_defaults_to_str(self) -> None:
        t = _resolve_type({"type": "array"}, {}, "f", 0)
        assert get_origin(t) is list
        assert get_args(t) == (str,)

    def test_array_of_nested_objects(self) -> None:
        prop: dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
        }
        t = _resolve_type(prop, {}, "items", 0)
        assert get_origin(t) is list
        inner = get_args(t)[0]
        assert issubclass(inner, BaseModel)


# ---------------------------------------------------------------------------
# _resolve_type — 嵌套 object
# ---------------------------------------------------------------------------
class TestResolveTypeObject:
    def test_object_with_properties(self) -> None:
        prop: dict[str, Any] = {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "zip_code": {"type": "string"},
            },
        }
        model = _resolve_type(prop, {}, "address", 0)
        assert issubclass(model, BaseModel)
        assert "street" in model.model_fields
        assert "zip_code" in model.model_fields

    def test_object_without_properties_returns_dict(self) -> None:
        t = _resolve_type({"type": "object"}, {}, "data", 0)
        assert get_origin(t) is dict

    def test_deeply_nested_object(self) -> None:
        prop: dict[str, Any] = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                    },
                },
            },
        }
        model = _resolve_type(prop, {}, "outer", 0)
        assert issubclass(model, BaseModel)
        inner_field = model.model_fields["inner"]
        assert issubclass(inner_field.annotation, BaseModel)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _resolve_type — enum
# ---------------------------------------------------------------------------
class TestResolveTypeEnum:
    def test_string_enum(self) -> None:
        t = _resolve_type({"enum": ["red", "green", "blue"]}, {}, "color", 0)
        assert get_origin(t) is Literal
        assert set(get_args(t)) == {"red", "green", "blue"}

    def test_single_value_enum(self) -> None:
        t = _resolve_type({"enum": ["only"]}, {}, "f", 0)
        assert get_origin(t) is Literal
        assert get_args(t) == ("only",)

    def test_mixed_type_enum(self) -> None:
        t = _resolve_type({"enum": ["a", 1, True]}, {}, "f", 0)
        assert get_origin(t) is Literal

    def test_empty_enum_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _resolve_type({"enum": []}, {}, "f", 0)
        assert "enum 不能为空" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# _resolve_type — oneOf / anyOf
# ---------------------------------------------------------------------------
class TestResolveTypeUnion:
    def test_oneof_primitive(self) -> None:
        prop: dict[str, Any] = {
            "oneOf": [{"type": "string"}, {"type": "integer"}],
        }
        t = _resolve_type(prop, {}, "f", 0)
        assert get_origin(t) is Union
        assert set(get_args(t)) == {str, int}

    def test_anyof_primitive(self) -> None:
        prop: dict[str, Any] = {
            "anyOf": [{"type": "number"}, {"type": "boolean"}],
        }
        t = _resolve_type(prop, {}, "f", 0)
        assert get_origin(t) is Union
        assert set(get_args(t)) == {float, bool}

    def test_single_item_oneof_unwraps(self) -> None:
        prop: dict[str, Any] = {"oneOf": [{"type": "string"}]}
        t = _resolve_type(prop, {}, "f", 0)
        assert t is str

    def test_oneof_with_ref(self) -> None:
        root: dict[str, Any] = {
            "$defs": {
                "Dog": {"type": "object", "properties": {"bark": {"type": "boolean"}}},
                "Cat": {"type": "object", "properties": {"meow": {"type": "boolean"}}},
            },
        }
        prop: dict[str, Any] = {
            "oneOf": [
                {"$ref": "#/$defs/Dog"},
                {"$ref": "#/$defs/Cat"},
            ],
        }
        t = _resolve_type(prop, root, "pet", 0)
        assert get_origin(t) is Union
        args = get_args(t)
        assert len(args) == 2
        assert all(issubclass(a, BaseModel) for a in args)

    def test_anyof_with_enum_and_type(self) -> None:
        prop: dict[str, Any] = {
            "anyOf": [
                {"enum": ["auto", "manual"]},
                {"type": "integer"},
            ],
        }
        t = _resolve_type(prop, {}, "mode", 0)
        assert get_origin(t) is Union

    def test_empty_oneof_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _resolve_type({"oneOf": []}, {}, "f", 0)
        assert "oneOf" in str(exc_info.value.detail)

    def test_empty_anyof_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _resolve_type({"anyOf": []}, {}, "f", 0)
        assert "anyOf" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# _resolve_type — $ref
# ---------------------------------------------------------------------------
class TestResolveTypeRef:
    def test_ref_to_primitive(self) -> None:
        root: dict[str, Any] = {"$defs": {"MyStr": {"type": "string"}}}
        t = _resolve_type({"$ref": "#/$defs/MyStr"}, root, "f", 0)
        assert t is str

    def test_ref_to_object(self) -> None:
        root: dict[str, Any] = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                },
            },
        }
        t = _resolve_type({"$ref": "#/$defs/Address"}, root, "addr", 0)
        assert issubclass(t, BaseModel)
        assert "city" in t.model_fields

    def test_ref_in_array_items(self) -> None:
        root: dict[str, Any] = {
            "$defs": {
                "Tag": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                },
            },
        }
        prop: dict[str, Any] = {
            "type": "array",
            "items": {"$ref": "#/$defs/Tag"},
        }
        t = _resolve_type(prop, root, "tags", 0)
        assert get_origin(t) is list
        inner = get_args(t)[0]
        assert issubclass(inner, BaseModel)

    def test_ref_uses_definitions_key(self) -> None:
        root: dict[str, Any] = {
            "definitions": {"Score": {"type": "integer"}},
        }
        t = _resolve_type({"$ref": "#/definitions/Score"}, root, "f", 0)
        assert t is int


# ---------------------------------------------------------------------------
# _resolve_type — 递归深度限制
# ---------------------------------------------------------------------------
class TestResolveTypeDepthLimit:
    def test_exceeds_max_depth_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _resolve_type({"type": "string"}, {}, "f", depth=11)
        assert "嵌套层级超过上限" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# _build_object_model — description 透传
# ---------------------------------------------------------------------------
class TestBuildObjectModelDescription:
    def test_description_propagated_to_field(self) -> None:
        schema: dict[str, Any] = {
            "properties": {
                "name": {"type": "string", "description": "用户姓名"},
                "age": {"type": "integer", "description": "用户年龄"},
            },
        }
        model = _build_object_model(schema, schema, "User", depth=0)
        assert model.model_fields["name"].description == "用户姓名"
        assert model.model_fields["age"].description == "用户年龄"

    def test_no_description_field_has_none_description(self) -> None:
        schema: dict[str, Any] = {
            "properties": {"x": {"type": "integer"}},
        }
        model = _build_object_model(schema, schema, "M", depth=0)
        assert model.model_fields["x"].description is None

    def test_description_with_default(self) -> None:
        schema: dict[str, Any] = {
            "properties": {
                "color": {
                    "type": "string",
                    "default": "red",
                    "description": "主题颜色",
                },
            },
        }
        model = _build_object_model(schema, schema, "Theme", depth=0)
        field = model.model_fields["color"]
        assert field.description == "主题颜色"
        assert field.default == "red"

    def test_description_coexists_with_ref(self) -> None:
        root: dict[str, Any] = {
            "$defs": {
                "Addr": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
            "properties": {
                "home": {
                    "$ref": "#/$defs/Addr",
                    "description": "家庭住址",
                },
            },
        }
        model = _build_object_model(root, root, "Person", depth=0)
        assert model.model_fields["home"].description == "家庭住址"


# ---------------------------------------------------------------------------
# _build_object_model — 字段数上限
# ---------------------------------------------------------------------------
class TestBuildObjectModelLimits:
    def test_exceeds_max_properties_raises(self) -> None:
        from fastapi import HTTPException

        props = {f"f{i}": {"type": "string"} for i in range(51)}
        with pytest.raises(HTTPException) as exc_info:
            _build_object_model({"properties": props}, {}, "M", 0)
        assert "不能超过 50" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# _schema_to_model 集成测试
# ---------------------------------------------------------------------------
class TestSchemaToModel:
    def test_basic_types(self) -> None:
        schema: dict[str, Any] = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
            },
        }
        model = _schema_to_model(schema)
        assert issubclass(model, BaseModel)
        fields = model.model_fields
        assert "name" in fields
        assert "age" in fields

    def test_empty_properties_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            _schema_to_model({"properties": {}})

    def test_no_properties_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            _schema_to_model({})

    def test_nested_object(self) -> None:
        schema: dict[str, Any] = {
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "zip": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
        model = _schema_to_model(schema)
        user_type = model.model_fields["user"].annotation
        assert issubclass(user_type, BaseModel)
        addr_type = user_type.model_fields["address"].annotation
        assert issubclass(addr_type, BaseModel)

    def test_enum_constraint(self) -> None:
        schema: dict[str, Any] = {
            "properties": {
                "status": {"enum": ["active", "inactive", "pending"]},
            },
        }
        model = _schema_to_model(schema)
        field_type = model.model_fields["status"].annotation
        assert get_origin(field_type) is Literal

    def test_ref_schema(self) -> None:
        schema: dict[str, Any] = {
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                },
            },
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Item"},
                },
            },
        }
        model = _schema_to_model(schema)
        field_type = model.model_fields["items"].annotation
        assert get_origin(field_type) is list
        inner = get_args(field_type)[0]
        assert issubclass(inner, BaseModel)
        assert "id" in inner.model_fields

    def test_oneof_union(self) -> None:
        schema: dict[str, Any] = {
            "$defs": {
                "TextBlock": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
                "ImageBlock": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                },
            },
            "properties": {
                "content": {
                    "oneOf": [
                        {"$ref": "#/$defs/TextBlock"},
                        {"$ref": "#/$defs/ImageBlock"},
                    ],
                },
            },
        }
        model = _schema_to_model(schema)
        field_type = model.model_fields["content"].annotation
        assert get_origin(field_type) is Union

    def test_description_passthrough(self) -> None:
        schema: dict[str, Any] = {
            "properties": {
                "title": {
                    "type": "string",
                    "description": "文章标题",
                },
                "summary": {
                    "type": "string",
                    "description": "摘要内容",
                },
            },
        }
        model = _schema_to_model(schema)
        assert model.model_fields["title"].description == "文章标题"
        assert model.model_fields["summary"].description == "摘要内容"

    def test_complex_real_world_schema(self) -> None:
        """模拟真实场景的复杂 Schema。"""
        schema: dict[str, Any] = {
            "$defs": {
                "Author": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "作者姓名"},
                        "role": {"enum": ["主作者", "联合作者", "通讯作者"]},
                    },
                },
            },
            "properties": {
                "title": {"type": "string", "description": "论文标题"},
                "authors": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Author"},
                    "description": "作者列表",
                },
                "year": {"type": "integer", "description": "发表年份"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词",
                },
                "category": {
                    "anyOf": [
                        {"enum": ["journal", "conference", "preprint"]},
                        {"type": "string"},
                    ],
                    "description": "论文分类",
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "doi": {"type": "string"},
                        "pages": {"type": "integer"},
                    },
                    "description": "元数据",
                },
            },
        }
        model = _schema_to_model(schema)
        assert issubclass(model, BaseModel)
        fields = model.model_fields
        assert fields["title"].description == "论文标题"
        assert fields["authors"].description == "作者列表"
        assert fields["category"].description == "论文分类"
        assert fields["metadata"].description == "元数据"

        author_type = get_args(fields["authors"].annotation)[0]
        assert issubclass(author_type, BaseModel)
        assert "name" in author_type.model_fields

    def test_model_instantiation_basic(self) -> None:
        """验证生成的 Model 可以正常实例化与序列化。"""
        schema: dict[str, Any] = {
            "properties": {
                "name": {"type": "string", "description": "名称"},
                "score": {"type": "number", "default": 0.0},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        model = _schema_to_model(schema)
        instance = model(name="test", tags=["a", "b"])
        data = instance.model_dump()
        assert data["name"] == "test"
        assert data["score"] == 0.0
        assert data["tags"] == ["a", "b"]

    def test_model_instantiation_enum(self) -> None:
        schema: dict[str, Any] = {
            "properties": {
                "color": {"enum": ["red", "green", "blue"]},
            },
        }
        model = _schema_to_model(schema)
        instance = model(color="red")
        assert instance.model_dump()["color"] == "red"

    def test_model_instantiation_nested(self) -> None:
        schema: dict[str, Any] = {
            "properties": {
                "addr": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                },
            },
        }
        model = _schema_to_model(schema)
        instance = model(addr={"city": "北京"})
        assert instance.model_dump()["addr"]["city"] == "北京"

    def test_model_validation_enum_rejects_invalid(self) -> None:
        """enum 约束应拒绝非法值。"""
        from pydantic import ValidationError

        schema: dict[str, Any] = {
            "properties": {
                "status": {"enum": ["a", "b"]},
            },
        }
        model = _schema_to_model(schema)
        with pytest.raises(ValidationError):
            model(status="invalid_value")
