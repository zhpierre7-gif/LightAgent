#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: [weego/WXAI-Team]
Last updated: 2026-02-22
"""

import os
import yaml
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging


@dataclass
class Skill:
    """Skill data class."""
    name: str
    description: str
    path: str
    instructions: str = ""
    has_scripts: bool = False
    has_references: bool = False
    has_assets: bool = False


class SkillManager:
    """Skill manager: discovers, loads, and executes skills."""

    def __init__(self, skills_directories: List[str] = None, logger=None):
        self.skills_directories = skills_directories or ["skills"]
        self.skills: Dict[str, Skill] = {}
        self.logger = logger or logging.getLogger(__name__)

    def discover_skills(self) -> List[Skill]:
        """Discover all available skills (metadata only)."""
        discovered = []

        for base_dir in self.skills_directories:
            if not os.path.exists(base_dir):
                continue

            for item in os.listdir(base_dir):
                skill_path = os.path.join(base_dir, item)
                skill_file = os.path.join(skill_path, "SKILL.md")

                if os.path.isdir(skill_path) and os.path.exists(skill_file):
                    try:
                        skill = self._load_skill_metadata(skill_path)
                        if skill:
                            self.skills[skill.name] = skill
                            discovered.append(skill)
                            self._log("DEBUG", "discover_skill",
                                      {"name": skill.name, "path": skill_path})
                    except Exception as e:
                        self._log("ERROR", "discover_skill_failed",
                                  {"path": skill_path, "error": str(e)})

        return discovered

    def _load_skill_metadata(self, skill_path: str) -> Optional[Skill]:
        """Load skill metadata from SKILL.md (frontmatter only)."""
        skill_file = os.path.join(skill_path, "SKILL.md")

        with open(skill_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse YAML frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if not frontmatter_match:
            raise ValueError(f"Missing frontmatter in {skill_file}")

        frontmatter = yaml.safe_load(frontmatter_match.group(1))

        # Validate required fields
        if 'name' not in frontmatter or 'description' not in frontmatter:
            raise ValueError(f"Skill missing required fields (name, description)")

        # Check for optional directories
        has_scripts = os.path.exists(os.path.join(skill_path, "scripts"))
        has_references = os.path.exists(os.path.join(skill_path, "references"))
        has_assets = os.path.exists(os.path.join(skill_path, "assets"))

        return Skill(
            name=frontmatter['name'],
            description=frontmatter['description'],
            path=skill_path,
            has_scripts=has_scripts,
            has_references=has_references,
            has_assets=has_assets
        )

    def activate_skill(self, skill_name: str) -> str:
        """Activate a skill: load its full instructions into context."""
        if skill_name not in self.skills:
            raise ValueError(f"Skill '{skill_name}' not found")

        skill = self.skills[skill_name]
        skill_file = os.path.join(skill.path, "SKILL.md")

        with open(skill_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Strip frontmatter and keep only the instructions body
        content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
        skill.instructions = content.strip()

        self._log("INFO", "activate_skill", {"name": skill_name})
        return skill.instructions

    def get_skills_xml(self) -> str:
        """Generate skill metadata in XML format (for use in system prompts)."""
        if not self.skills:
            return ""

        xml_parts = ['<available_skills>']

        for skill in self.skills.values():
            xml_parts.append(f'  <skill>')
            xml_parts.append(f'    <name>{skill.name}</name>')
            xml_parts.append(f'    <description>{skill.description}</description>')
            xml_parts.append(f'    <location>{os.path.join(skill.path, "SKILL.md")}</location>')
            xml_parts.append(f'  </skill>')

        xml_parts.append('</available_skills>')
        return '\n'.join(xml_parts)

    def execute_script(self, skill_name: str, script_name: str, args: List[str] = None) -> str:
        """Execute a script from within a skill (sandboxed)."""
        if skill_name not in self.skills:
            return f"Error: Skill '{skill_name}' not found"

        skill = self.skills[skill_name]
        script_path = os.path.join(skill.path, "scripts", script_name)

        if not os.path.exists(script_path):
            return f"Error: Script '{script_path}' not found in skill '{skill_name}'"

        # Security check: only allow execution of files inside the scripts directory
        if not script_path.startswith(os.path.join(skill.path, "scripts")):
            return "Error: Security violation - cannot execute outside scripts directory"

        try:
            # Run in a temporary directory for isolation
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    [script_path] + (args or []),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir
                )

                output = result.stdout
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"

                self._log("DEBUG", "execute_script",
                          {"skill": skill_name, "script": script_name, "output": output[:200]})
                return output

        except subprocess.TimeoutExpired:
            return "Error: Script execution timeout (30s)"
        except Exception as e:
            return f"Error executing script: {str(e)}"

    def read_reference(self, skill_name: str, ref_path: str) -> str:
        """Read a reference document from within a skill."""
        if skill_name not in self.skills:
            return f"Error: Skill '{skill_name}' not found"

        skill = self.skills[skill_name]
        full_path = os.path.join(skill.path, "references", ref_path)

        # Security check: prevent directory traversal
        if not full_path.startswith(os.path.join(skill.path, "references")):
            return "Error: Security violation - invalid reference path"

        if not os.path.exists(full_path):
            return f"Error: Reference '{ref_path}' not found"

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading reference: {str(e)}"

    def read_asset(self, skill_name: str, asset_path: str) -> bytes:
        """Read an asset file from within a skill (returns raw bytes)."""
        if skill_name not in self.skills:
            raise ValueError(f"Skill '{skill_name}' not found")

        skill = self.skills[skill_name]
        full_path = os.path.join(skill.path, "assets", asset_path)

        # Security check
        if not full_path.startswith(os.path.join(skill.path, "assets")):
            raise ValueError("Security violation: invalid asset path")

        with open(full_path, 'rb') as f:
            return f.read()

    def _log(self, level: str, action: str, data: Any):
        """Unified logging method, compatible with LightAgent's LoggerManager and standard logging."""
        if not self.logger:
            return

        from .logger import LoggerManager

        if isinstance(self.logger, LoggerManager):
            self.logger.log(level, action, data)
        elif hasattr(self.logger, 'debug') and hasattr(self.logger, 'info') and hasattr(self.logger, 'error'):
            log_msg = f"[SkillManager] {action}: {data}"
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
            }
            self.logger.log(level_map.get(level, logging.INFO), log_msg)
        # If no suitable logger is available, silently skip logging
