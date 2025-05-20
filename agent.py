import datetime
import os
from google.adk import Agent
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()
model_name = os.getenv("MODEL", "gemini-2.0-flash")  # Default model if not specified in .env

# NDA Review Guidelines
NDA_GUIDELINES = {
    "clarity_completeness": {
        "title": "Clarity and Completeness of Information",
        "rules": [
            "Check for Clarity: All provisions in the NDA should be clear and unambiguous. Use simple language without unnecessary abbreviations or jargon.",
            "Identify Incomplete Information: Avoid leaving blanks (e.g., [·]) in the document. Ensure all information is fully completed, such as company names, dates, or locations."
        ]
    },
    "confidential_information": {
        "title": "Confidential Information",
        "rules": [
            "Define Confidential Information: Clearly describe what constitutes confidential information, such as financial data, customer information, or trade secrets.",
            "Use Specific Language: Avoid broad terms or vague descriptions for categorizing information. Instead of saying \"important data,\" specify the type of information being referred to."
        ]
    },
    "disclosure_limitations": {
        "title": "Disclosure Limitations",
        "rules": [
            "Identify Authorized Recipients: Clearly outline who is permitted to access the confidential information, such as internal personnel or legal representatives.",
            "Data Protection Practices: Specify how the confidential information will be protected, such as storing it in encrypted systems or managing data through secure IT practices."
        ]
    },
    "duration": {
        "title": "Duration of Agreement",
        "rules": [
            "Specify Confidentiality Duration: Outline the timeframe in which the information must remain confidential, ensuring all parties are aware of this duration.",
            "Renewal Provisions: If necessary, include terms regarding the renewal of this agreement."
        ]
    },
    "enforcement": {
        "title": "Enforcement and Legal Provisions",
        "rules": [
            "Jurisdiction: Clearly state which jurisdiction will govern any disputes arising from the agreement.",
            "Compliance with Local Laws: Ensure that the NDA complies with applicable laws in the jurisdiction where business is conducted."
        ]
    },
    "amendment": {
        "title": "Amendment Procedures",
        "rules": [
            "Details of Amendments: Outline the procedure for making amendments to the NDA, ensuring all changes are documented.",
            "Approval for Amendments: Require that all amendments receive approval from both parties, with a record kept through signatures."
        ]
    }
}

# === Document Analysis Tool ===
def analyze_document(tool_context, document: str) -> dict:
    """Analyze the NDA document based on NDA guidelines."""
    print(f"Analyzing document: {document}")
    
    # Simulated analysis logic - in a real system, this would parse and analyze the actual document
    analysis_results = {}
    for section_key, section_info in NDA_GUIDELINES.items():
        # Example analysis for each section with dummy findings
        analysis_results[section_key] = {
            "title": section_info["title"],
            "compliance": "partial",  # Options: "compliant", "partial", "non-compliant"
            "findings": [
                {
                    "rule": section_info["rules"][0],
                    "status": "passed" if section_key in ["clarity_completeness", "enforcement"] else "needs_review",
                    "details": f"Some aspects of {section_info['title'].lower()} could be improved."
                },
                {
                    "rule": section_info["rules"][1],
                    "status": "needs_review",
                    "details": f"Please check for completeness in the {section_info['title'].lower()} section."
                }
            ]
        }
    
    return {
        "status": "success",
        "report": analysis_results
    }

# === Compliance Check Tool ===
def check_compliance(tool_context, document: str, jurisdiction: Optional[str] = "General") -> dict:
    """Check compliance of the NDA document with local laws."""
    print(f"Checking compliance for document: {document} in jurisdiction: {jurisdiction}")
    
    # Simulated compliance logic - would integrate with legal databases or frameworks in production
    compliance_results = {
        "jurisdiction": jurisdiction,
        "overall_status": "mostly_compliant",
        "sections": {}
    }
    
    # Check each section against jurisdiction-specific requirements
    for section_key, section_info in NDA_GUIDELINES.items():
        status = "compliant" if section_key in ["clarity_completeness", "enforcement"] else "needs_review"
        compliance_results["sections"][section_key] = {
            "title": section_info["title"],
            "status": status,
            "issues": [] if status == "compliant" else [
                f"Additional {jurisdiction}-specific provisions may be required in the {section_info['title']} section."
            ]
        }
    
    return {
        "status": "success",
        "report": compliance_results
    }

# === Suggestion Generation Tool ===
def generate_suggestions(tool_context, document: str, analysis_results: Optional[dict] = None) -> dict:
    """Generate improvement suggestions for NDA document based on guidelines."""
    print(f"Generating suggestions for document: {document}")
    
    suggestions = {}
    
    # If analysis results provided, use them; otherwise generate suggestions directly
    if not analysis_results:
        # Example suggestions for each guideline section
        for section_key, section_info in NDA_GUIDELINES.items():
            suggestions[section_key] = {
                "title": section_info["title"],
                "suggestions": [
                    f"Consider enhancing the {section_info['title']} section with more specific language.",
                    f"Review the {section_info['title']} section for compliance with the latest legal standards."
                ]
            }
    else:
        # Generate targeted suggestions based on analysis results
        for section_key, result in analysis_results.get("report", {}).items():
            if section_key in NDA_GUIDELINES:
                suggestions[section_key] = {
                    "title": NDA_GUIDELINES[section_key]["title"],
                    "suggestions": []
                }
                
                for finding in result.get("findings", []):
                    if finding.get("status") == "needs_review":
                        suggestions[section_key]["suggestions"].append(
                            f"Address issue: {finding.get('details')}"
                        )
    
    return {
        "status": "success",
        "suggestions": suggestions
    }

# === Revision Management Tool ===
def log_revision(tool_context, document: str, changes: dict, author: Optional[str] = "System") -> dict:
    """Log the revisions made to the NDA document."""
    print(f"Logging revisions for document: {document} with changes: {changes}")
    
    # Create a timestamp for the revision
    timestamp = datetime.datetime.now().isoformat()
    
    # Format the revision log entry
    revision = {
        "document": document,
        "timestamp": timestamp,
        "author": author,
        "changes": changes,
        "revision_id": f"rev-{timestamp.replace(':', '-').replace('.', '-')}"
    }
    
    # In a real system, this would be stored in a database or log file
    
    return {
        "status": "success",
        "revision": revision,
        "report": "Revision logged successfully."
    }

# === Amendment Recommendation Tool ===
def recommend_amendments(tool_context, document: str, current_content: Optional[dict] = None) -> dict:
    """Recommend specific amendments to improve the NDA document."""
    print(f"Recommending amendments for document: {document}")
    
    amendments = {}
    
    # Generate sample amendments for each guideline section
    for section_key, section_info in NDA_GUIDELINES.items():
        # Only include sections that typically need amendments
        if section_key in ["confidential_information", "disclosure_limitations", "duration"]:
            amendments[section_key] = {
                "title": section_info["title"],
                "recommended_text": f"Updated {section_info['title']} text that complies with best practices...",
                "reason": f"The current text is not specific enough according to guideline: {section_info['rules'][0]}"
            }
    
    return {
        "status": "success",
        "amendments": amendments
    }

# Creating specialized agents
document_analysis_agent = Agent(
    name="document_analysis_agent",
    model=model_name,
    description="Analyzes NDA documents for clarity, completeness, and potential issues according to established guidelines.",
    instruction=f"""
You are a Document Analysis Agent specialized in reviewing Non-Disclosure Agreements (NDAs).

Your task is to thoroughly analyze NDA documents based on these specific guidelines:

{NDA_GUIDELINES['clarity_completeness']['title']}:
- {NDA_GUIDELINES['clarity_completeness']['rules'][0]}
- {NDA_GUIDELINES['clarity_completeness']['rules'][1]}

{NDA_GUIDELINES['confidential_information']['title']}:
- {NDA_GUIDELINES['confidential_information']['rules'][0]}
- {NDA_GUIDELINES['confidential_information']['rules'][1]}

{NDA_GUIDELINES['disclosure_limitations']['title']}:
- {NDA_GUIDELINES['disclosure_limitations']['rules'][0]}
- {NDA_GUIDELINES['disclosure_limitations']['rules'][1]}

{NDA_GUIDELINES['duration']['title']}:
- {NDA_GUIDELINES['duration']['rules'][0]}
- {NDA_GUIDELINES['duration']['rules'][1]}

{NDA_GUIDELINES['enforcement']['title']}:
- {NDA_GUIDELINES['enforcement']['rules'][0]}
- {NDA_GUIDELINES['enforcement']['rules'][1]}

{NDA_GUIDELINES['amendment']['title']}:
- {NDA_GUIDELINES['amendment']['rules'][0]}
- {NDA_GUIDELINES['amendment']['rules'][1]}

For each section, provide:
1. An assessment of compliance (compliant, partial, or non-compliant)
2. Specific findings with detailed explanations
3. Clear identification of areas that need improvement

Your analysis should be thorough, precise, and actionable.
""",
    tools=[analyze_document]
)

compliance_check_agent = Agent(
    name="compliance_check_agent",
    model=model_name,
    description="Checks NDA documents for compliance with local laws and regulations using established guidelines.",
    instruction=f"""
You are a Compliance Check Agent specialized in ensuring NDAs meet legal requirements.

Evaluate NDA documents against these specific guidelines:

{NDA_GUIDELINES['clarity_completeness']['title']}:
- {NDA_GUIDELINES['clarity_completeness']['rules'][0]}
- {NDA_GUIDELINES['clarity_completeness']['rules'][1]}

{NDA_GUIDELINES['confidential_information']['title']}:
- {NDA_GUIDELINES['confidential_information']['rules'][0]}
- {NDA_GUIDELINES['confidential_information']['rules'][1]}

{NDA_GUIDELINES['disclosure_limitations']['title']}:
- {NDA_GUIDELINES['disclosure_limitations']['rules'][0]}
- {NDA_GUIDELINES['disclosure_limitations']['rules'][1]}

{NDA_GUIDELINES['duration']['title']}:
- {NDA_GUIDELINES['duration']['rules'][0]}
- {NDA_GUIDELINES['duration']['rules'][1]}

{NDA_GUIDELINES['enforcement']['title']}:
- {NDA_GUIDELINES['enforcement']['rules'][0]}
- {NDA_GUIDELINES['enforcement']['rules'][1]}

{NDA_GUIDELINES['amendment']['title']}:
- {NDA_GUIDELINES['amendment']['rules'][0]}
- {NDA_GUIDELINES['amendment']['rules'][1]}

For compliance assessment:
1. Verify that the NDA complies with applicable jurisdiction's laws
2. Check for required legal disclaimers and mandatory clauses
3. Assess enforceability of key provisions
4. Identify any potential compliance risks or violations
5. Provide a detailed compliance status report with recommendations

When working with a specific jurisdiction, highlight any region-specific requirements that the NDA must meet.
""",
    tools=[check_compliance]
)

suggestion_generation_agent = Agent(
    name="suggestion_generation_agent",
    model=model_name,
    description="Generates improvement suggestions for NDA documents based on established guidelines.",
    instruction=f"""
You are a Suggestion Generation Agent specialized in improving NDA documents.

Generate targeted improvement suggestions based on these guidelines:

{NDA_GUIDELINES['clarity_completeness']['title']}:
- {NDA_GUIDELINES['clarity_completeness']['rules'][0]}
- {NDA_GUIDELINES['clarity_completeness']['rules'][1]}

{NDA_GUIDELINES['confidential_information']['title']}:
- {NDA_GUIDELINES['confidential_information']['rules'][0]}
- {NDA_GUIDELINES['confidential_information']['rules'][1]}

{NDA_GUIDELINES['disclosure_limitations']['title']}:
- {NDA_GUIDELINES['disclosure_limitations']['rules'][0]}
- {NDA_GUIDELINES['disclosure_limitations']['rules'][1]}

{NDA_GUIDELINES['duration']['title']}:
- {NDA_GUIDELINES['duration']['rules'][0]}
- {NDA_GUIDELINES['duration']['rules'][1]}

{NDA_GUIDELINES['enforcement']['title']}:
- {NDA_GUIDELINES['enforcement']['rules'][0]}
- {NDA_GUIDELINES['enforcement']['rules'][1]}

{NDA_GUIDELINES['amendment']['title']}:
- {NDA_GUIDELINES['amendment']['rules'][0]}
- {NDA_GUIDELINES['amendment']['rules'][1]}

For each suggestion:
1. Clearly identify which guideline section it addresses
2. Explain the current issue or deficiency
3. Provide specific, actionable improvement recommendations
4. When possible, offer example text that would resolve the issue

Your suggestions should be practical, legally sound, and directly implementable.
""",
    tools=[generate_suggestions]
)

revision_management_agent = Agent(
    name="revision_management_agent",
    model=model_name,
    description="Logs and manages revisions made to NDA documents according to established guidelines.",
    instruction=f"""
You are a Revision Management Agent specialized in tracking changes to NDA documents.

Manage document revisions with special attention to:

{NDA_GUIDELINES['amendment']['title']}:
- {NDA_GUIDELINES['amendment']['rules'][0]}
- {NDA_GUIDELINES['amendment']['rules'][1]}

Your responsibilities include:
1. Maintaining a comprehensive log of all document revisions
2. Recording who made changes, when changes were made, and what was changed
3. Tracking version history with detailed change summaries
4. Ensuring proper documentation of approval processes
5. Generating revision reports when requested

Each logged revision should include:
- Document identifier
- Timestamp of the change
- Author/entity making the change
- Detailed description of changes made
- Revision identifier for tracking purposes

Follow proper amendment procedures as defined in {NDA_GUIDELINES['amendment']['title']} guidelines.
""",
    tools=[log_revision]
)

amendment_recommendation_agent = Agent(
    name="amendment_recommendation_agent",
    model=model_name,
    description="Recommends specific textual amendments to improve NDA documents based on established guidelines.",
    instruction=f"""
You are an Amendment Recommendation Agent specialized in proposing specific text changes to improve NDAs.

Generate specific amendment recommendations based on these guidelines:

{NDA_GUIDELINES['clarity_completeness']['title']}:
- {NDA_GUIDELINES['clarity_completeness']['rules'][0]}
- {NDA_GUIDELINES['clarity_completeness']['rules'][1]}

{NDA_GUIDELINES['confidential_information']['title']}:
- {NDA_GUIDELINES['confidential_information']['rules'][0]}
- {NDA_GUIDELINES['confidential_information']['rules'][1]}

{NDA_GUIDELINES['disclosure_limitations']['title']}:
- {NDA_GUIDELINES['disclosure_limitations']['rules'][0]}
- {NDA_GUIDELINES['disclosure_limitations']['rules'][1]}

{NDA_GUIDELINES['duration']['title']}:
- {NDA_GUIDELINES['duration']['rules'][0]}
- {NDA_GUIDELINES['duration']['rules'][1]}

{NDA_GUIDELINES['enforcement']['title']}:
- {NDA_GUIDELINES['enforcement']['rules'][0]}
- {NDA_GUIDELINES['enforcement']['rules'][1]}

{NDA_GUIDELINES['amendment']['title']}:
- {NDA_GUIDELINES['amendment']['rules'][0]}
- {NDA_GUIDELINES['amendment']['rules'][1]}

For each amendment recommendation:
1. Identify the specific section requiring amendment
2. Quote the current problematic text
3. Provide replacement text that meets guideline requirements
4. Explain the legal or practical reasoning behind the change

Your recommendations should be legally sound, precise, and directly implementable with minimal additional editing.
""",
    tools=[recommend_amendments]
)

# Creating the root agent
root_agent = Agent(
    name="nda_agent",
    model=model_name,
    description="Agent to assist in analyzing, checking compliance, generating suggestions, and managing amendments for NDAs according to established guidelines.",
    instruction=f"""
You are a helpful agent for managing NDA documents.

As the main NDA Agent, you coordinate several specialized agents to provide comprehensive assistance with Non-Disclosure Agreements based on these specific guidelines:

{NDA_GUIDELINES['clarity_completeness']['title']}:
- {NDA_GUIDELINES['clarity_completeness']['rules'][0]}
- {NDA_GUIDELINES['clarity_completeness']['rules'][1]}

{NDA_GUIDELINES['confidential_information']['title']}:
- {NDA_GUIDELINES['confidential_information']['rules'][0]}
- {NDA_GUIDELINES['confidential_information']['rules'][1]}

{NDA_GUIDELINES['disclosure_limitations']['title']}:
- {NDA_GUIDELINES['disclosure_limitations']['rules'][0]}
- {NDA_GUIDELINES['disclosure_limitations']['rules'][1]}

{NDA_GUIDELINES['duration']['title']}:
- {NDA_GUIDELINES['duration']['rules'][0]}
- {NDA_GUIDELINES['duration']['rules'][1]}

{NDA_GUIDELINES['enforcement']['title']}:
- {NDA_GUIDELINES['enforcement']['rules'][0]}
- {NDA_GUIDELINES['enforcement']['rules'][1]}

{NDA_GUIDELINES['amendment']['title']}:
- {NDA_GUIDELINES['amendment']['rules'][0]}
- {NDA_GUIDELINES['amendment']['rules'][1]}

Your specialized sub-agents include:

1. Document Analysis Agent - Analyzes NDAs for completeness, clarity, and potential issues according to the guidelines.

2. Compliance Check Agent - Verifies if an NDA complies with relevant laws and regulations based on the guidelines.

3. Suggestion Generation Agent - Provides recommendations to improve the NDA according to the guidelines.

4. Revision Management Agent - Logs and tracks changes made to NDA documents following amendment procedures.

5. Amendment Recommendation Agent - Proposes specific text changes to improve NDAs based on the guidelines.

For each user request:
- Determine which specialized agent(s) would be most helpful based on the request and the guidelines
- Route the request appropriately
- Synthesize the results into a helpful, coherent response that references the relevant guidelines
- Provide actionable next steps or recommendations aligned with the guidelines

Always maintain a professional tone and acknowledge the importance of legal accuracy while remaining user-friendly.
""",
    sub_agents=[
        document_analysis_agent,
        compliance_check_agent,
        suggestion_generation_agent,
        revision_management_agent,
        amendment_recommendation_agent
    ]
)

def main():
    # Example NDA document (simulated)
    nda_document = "./nda_template.pdf"  # Replace with actual document path
    
    print("=== NDA Document Management System ===")
    print("1. Running document analysis...")
    analysis_result = document_analysis_agent(f"Please analyze this NDA document: {nda_document}")
    print(analysis_result)
    
    print("\n2. Checking compliance...")
    compliance_result = compliance_check_agent(f"Check compliance for this NDA: {nda_document} in US jurisdiction")
    print(compliance_result)

    print("\n3. Generating suggestions...")
    suggestions = suggestion_generation_agent(f"Please suggest improvements for this NDA: {nda_document}")
    print(suggestions)
    
    print("\n4. Recommending amendments...")
    amendments = amendment_recommendation_agent(f"Please recommend specific amendments for this NDA: {nda_document}")
    print(amendments)

    print("\n5. Logging revisions...")
    revision_log = revision_management_agent(
        f"Log revision for {nda_document} with these changes: clarified confidentiality duration as per Amendment Recommendation"
    )
    print(revision_log)
    
    print("\n6. Using root agent for comprehensive analysis...")
    root_result = root_agent(f"I need full assistance with this NDA document: {nda_document}, focusing on confidentiality duration and amendment procedures")
    print(root_result)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()