"""Memory-graph (entities and relations) endpoints.

Extracted from app/main.py (N2 phase 2). Bodies unchanged.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import ROOT
from app.deps import state

router = APIRouter(tags=["memory-graph"])

@router.get("/memory-graph")
async def memory_graph_page():
    template = ROOT / "app" / "templates" / "memory_graph.html"
    if not template.exists():
        raise HTTPException(status_code=404, detail="Memory graph template not found")
    return FileResponse(template)


@router.get("/memory-graph/data")
async def memory_graph_data():
    entities = state.db.list_entities()
    relations = state.db.list_relations(active_only=True)
    memories = state.db.list_memories()

    linked_memories: dict[int, list[dict]] = {int(e["id"]): [] for e in entities}
    for memory in memories:
        fact = memory.get("fact", "") or ""
        for entity in entities:
            entity_name = entity.get("name", "") or ""
            if entity_name.lower() in fact.lower():
                linked_memories[int(entity["id"])].append({
                    "id": memory["id"],
                    "fact": memory["fact"],
                    "category": memory["category"],
                })

    return {
        "entities": [
            {
                "id": int(entity["id"]),
                "label": entity["name"],
                "title": entity["name"],
                "type": entity["type"],
                "notes": entity.get("notes") or "",
                "linked_memories": linked_memories[int(entity["id"])],
            }
            for entity in entities
        ],
        "relations": [
            {
                "id": int(relation["id"]),
                "from": int(relation["source_id"]),
                "to": int(relation["target_id"]),
                "label": relation["relation"],
                "confidence": relation.get("confidence", 0.5),
            }
            for relation in relations
        ],
    }


@router.put("/memory-graph/entities/{entity_id}")
async def update_memory_graph_entity(entity_id: int, body: dict):
    name = (body.get("name") or "").strip()
    entity_type = (body.get("type") or "thing").strip()
    notes = (body.get("notes") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    updated = state.db.update_entity(entity_id, name=name, entity_type=entity_type, notes=notes)
    if not updated:
        raise HTTPException(status_code=404, detail="entity not found")
    return state.db.get_entity(entity_id)


@router.delete("/memory-graph/entities/{entity_id}")
async def delete_memory_graph_entity(entity_id: int):
    deleted = state.db.delete_entity(entity_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="entity not found")
    return {"deleted": entity_id}


@router.delete("/memory-graph/relations/{relation_id}")
async def delete_memory_graph_relation(relation_id: int):
    deleted = state.db.delete_relation(relation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="relation not found")
    return {"deleted": relation_id}
