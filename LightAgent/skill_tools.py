#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: [weego/WXAI-Team]
Last updated: 2026-02-20
"""

import json
from typing import Dict, Any, List


def create_skill_tools(skill_manager):
    """Create tool functions for interacting with skills."""

    def list_skills() -> str:
        """
        List all available skills.

        Returns:
            JSON string containing the list of skills.
        """
        skills = []
        for skill in skill_manager.skills.values():
            skills.append({
                "name": skill.name,
                "description": skill.description,
                "has_scripts": skill.has_scripts,
                "has_references": skill.has_references,
                "has_assets": skill.has_assets
            })
        return json.dumps({"skills": skills}, ensure_ascii=False, indent=2)

    # Attach tool_info attribute to the function
    list_skills.tool_info = {
        "tool_name": "list_skills",
        "tool_title": "List available skills",
        "tool_description": "Get a list of all loaded skills along with their descriptions.",
        "tool_params": []
    }

    def activate_skill(skill_name: str) -> str:
        """
        Activate the specified skill and retrieve its full instructions.

        Args:
            skill_name: Name of the skill.

        Returns:
            The skill's full instruction text.
        """
        try:
            instructions = skill_manager.activate_skill(skill_name)
            return instructions
        except Exception as e:
            return f"Failed to activate skill: {str(e)}"

    activate_skill.tool_info = {
        "tool_name": "activate_skill",
        "tool_title": "Activate skill",
        "tool_description": "Load the full instructions for the specified skill into context.",
        "tool_params": [
            {"name": "skill_name", "description": "Name of the skill to activate.", "type": "string", "required": True}
        ]
    }

    def execute_skill_script(skill_name: str, script_name: str, args: List[str] = None) -> str:
        """
        Execute a script from within a skill.

        Args:
            skill_name: Name of the skill.
            script_name: Script filename.
            args: List of arguments to pass to the script.

        Returns:
            Script execution output.
        """
        return skill_manager.execute_script(skill_name, script_name, args)

    execute_skill_script.tool_info = {
        "tool_name": "execute_skill_script",
        "tool_title": "Execute skill script",
        "tool_description": "Execute a script file found in the skill's scripts/ directory.",
        "tool_params": [
            {"name": "skill_name", "description": "Name of the skill.", "type": "string", "required": True},
            {"name": "script_name", "description": "Script filename.", "type": "string", "required": True},
            {"name": "args", "description": "List of arguments to pass to the script.", "type": "array", "required": False}
        ]
    }

    def read_skill_reference(skill_name: str, ref_path: str) -> str:
        """
        Read a reference document from within a skill.

        Args:
            skill_name: Name of the skill.
            ref_path: Path to the reference document (relative to the references/ directory).

        Returns:
            Document contents.
        """
        return skill_manager.read_reference(skill_name, ref_path)

    read_skill_reference.tool_info = {
        "tool_name": "read_skill_reference",
        "tool_title": "Read skill reference document",
        "tool_description": "Read the contents of a document in the skill's references/ directory.",
        "tool_params": [
            {"name": "skill_name", "description": "Name of the skill.", "type": "string", "required": True},
            {"name": "ref_path", "description": "Path to the reference document.", "type": "string", "required": True}
        ]
    }

    return [list_skills, activate_skill, execute_skill_script, read_skill_reference]