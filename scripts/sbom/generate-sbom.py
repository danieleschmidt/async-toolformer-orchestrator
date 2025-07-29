#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generator for async-toolformer-orchestrator

Generates SPDX-compliant SBOM for security and compliance tracking.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import tomllib
import uuid

def run_command(cmd: List[str]) -> str:
    """Run command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running {' '.join(cmd)}: {e}")
        return ""

def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    return {
        "commit": run_command(["git", "rev-parse", "HEAD"]),
        "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "origin": run_command(["git", "remote", "get-url", "origin"]),
        "commit_date": run_command(["git", "show", "-s", "--format=%ci", "HEAD"]),
    }

def get_dependencies() -> List[Dict[str, Any]]:
    """Extract dependencies from pyproject.toml and pip freeze."""
    dependencies = []
    
    # Read pyproject.toml
    try:
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
            
        # Main dependencies
        for dep in pyproject.get("project", {}).get("dependencies", []):
            dependencies.append(parse_dependency(dep, "runtime"))
            
        # Optional dependencies
        for group, deps in pyproject.get("project", {}).get("optional-dependencies", {}).items():
            for dep in deps:
                dependencies.append(parse_dependency(dep, f"optional-{group}"))
                
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
    
    # Get installed versions
    try:
        pip_list = run_command(["pip", "list", "--format=json"])
        installed = {pkg["name"].lower(): pkg["version"] for pkg in json.loads(pip_list)}
        
        # Update versions for installed packages
        for dep in dependencies:
            name_lower = dep["name"].lower()
            if name_lower in installed:
                dep["version"] = installed[name_lower]
                dep["installed"] = True
                
    except Exception as e:
        print(f"Error getting pip list: {e}")
    
    return dependencies

def parse_dependency(dep_string: str, dep_type: str) -> Dict[str, Any]:
    """Parse dependency string into structured format."""
    # Simple parsing - could be enhanced for complex version specs
    if ">=" in dep_string:
        name, version = dep_string.split(">=", 1)
        version_constraint = f">={version}"
    elif "==" in dep_string:
        name, version = dep_string.split("==", 1)
        version_constraint = f"=={version}"
    elif ">" in dep_string:
        name, version = dep_string.split(">", 1)
        version_constraint = f">{version}"
    else:
        name = dep_string
        version = None
        version_constraint = None
    
    return {
        "name": name.strip(),
        "version": version.strip() if version else None,
        "version_constraint": version_constraint,
        "type": dep_type,
        "installed": False,
        "spdx_id": f"SPDXRef-Package-{name.strip().replace('-', '').replace('_', '')}"
    }

def generate_spdx_sbom() -> Dict[str, Any]:
    """Generate SPDX 2.3 compliant SBOM."""
    git_info = get_git_info()
    dependencies = get_dependencies()
    
    # Read project info
    try:
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        project_info = pyproject.get("project", {})
    except:
        project_info = {}
    
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "documentName": f"{project_info.get('name', 'async-toolformer-orchestrator')}-SBOM",
        "documentNamespace": f"https://github.com/yourusername/async-toolformer-orchestrator/sbom/{uuid.uuid4()}",
        "creationInfo": {
            "created": datetime.utcnow().isoformat() + "Z",
            "creators": [
                "Tool: async-toolformer-sbom-generator",
                f"Organization: {project_info.get('authors', [{}])[0].get('name', 'Unknown')}"
            ],
            "licenseListVersion": "3.21"
        },
        "packages": [],
        "relationships": []
    }
    
    # Main package
    main_package = {
        "SPDXID": "SPDXRef-Package-AsyncToolformer",
        "name": project_info.get("name", "async-toolformer-orchestrator"),
        "downloadLocation": git_info.get("origin", "NOASSERTION"),
        "filesAnalyzed": False,
        "licenseConcluded": "MIT",
        "licenseDeclared": "MIT",
        "copyrightText": "NOASSERTION",
        "versionInfo": project_info.get("version", "0.1.0"),
        "supplier": f"Organization: {project_info.get('authors', [{}])[0].get('name', 'Unknown')}",
        "description": project_info.get("description", ""),
        "homepage": project_info.get("urls", {}).get("Repository", ""),
        "sourceInfo": f"Git commit: {git_info.get('commit', 'unknown')}",
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f"pkg:pypi/{project_info.get('name', 'async-toolformer-orchestrator')}@{project_info.get('version', '0.1.0')}"
            }
        ]
    }
    sbom["packages"].append(main_package)
    
    # Dependencies
    for dep in dependencies:
        if not dep["installed"]:
            continue
            
        dep_package = {
            "SPDXID": dep["spdx_id"],
            "name": dep["name"],
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION", 
            "copyrightText": "NOASSERTION",
            "versionInfo": dep["version"] or "unknown",
            "supplier": "NOASSERTION",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{dep['name']}@{dep['version'] or 'unknown'}"
                }
            ]
        }
        sbom["packages"].append(dep_package)
        
        # Add dependency relationship
        relationship_type = "DEPENDS_ON" if dep["type"] == "runtime" else "OPTIONAL_DEPENDENCY_OF"
        sbom["relationships"].append({
            "spdxElementId": "SPDXRef-Package-AsyncToolformer",
            "relationshipType": relationship_type,
            "relatedSpdxElement": dep["spdx_id"]
        })
    
    return sbom

def generate_cyclonedx_sbom() -> Dict[str, Any]:
    """Generate CycloneDX 1.5 compliant SBOM."""
    git_info = get_git_info()
    dependencies = get_dependencies()
    
    try:
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        project_info = pyproject.get("project", {})
    except:
        project_info = {}
    
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tools": [
                {
                    "vendor": "async-toolformer",
                    "name": "sbom-generator",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "library",
                "bom-ref": "async-toolformer-orchestrator",
                "name": project_info.get("name", "async-toolformer-orchestrator"),
                "version": project_info.get("version", "0.1.0"),
                "description": project_info.get("description", ""),
                "purl": f"pkg:pypi/{project_info.get('name', 'async-toolformer-orchestrator')}@{project_info.get('version', '0.1.0')}",
                "licenses": [{"license": {"id": "MIT"}}],
                "externalReferences": [
                    {
                        "type": "vcs",
                        "url": git_info.get("origin", "")
                    }
                ]
            }
        },
        "components": []
    }
    
    # Add dependencies as components
    for dep in dependencies:
        if not dep["installed"]:
            continue
            
        component = {
            "type": "library",
            "bom-ref": dep["name"],
            "name": dep["name"],
            "version": dep["version"] or "unknown",
            "purl": f"pkg:pypi/{dep['name']}@{dep['version'] or 'unknown'}",
            "scope": "required" if dep["type"] == "runtime" else "optional"
        }
        sbom["components"].append(component)
    
    return sbom

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python generate-sbom.py [spdx|cyclonedx] [output_file]")
        sys.exit(1)
    
    format_type = sys.argv[1].lower()
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if format_type == "spdx":
        sbom = generate_spdx_sbom()
        default_filename = "sbom.spdx.json"
    elif format_type == "cyclonedx":
        sbom = generate_cyclonedx_sbom()
        default_filename = "sbom.cyclonedx.json"
    else:
        print("Error: Format must be 'spdx' or 'cyclonedx'")
        sys.exit(1)
    
    output_path = output_file or default_filename
    
    with open(output_path, "w") as f:
        json.dump(sbom, f, indent=2)
    
    print(f"SBOM generated: {output_path}")
    print(f"Format: {format_type.upper()}")
    print(f"Packages: {len(sbom.get('packages', sbom.get('components', [])))}")

if __name__ == "__main__":
    main()