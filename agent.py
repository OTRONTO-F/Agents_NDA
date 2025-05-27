# Standard library imports
import os
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Any

# Third-party imports
import PyPDF2
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration constants
MODEL_NAME: str = os.getenv("MODEL", "gemini-2.0-flash")
CREDENTIALS_PATH: str = os.getenv(
    "GOOGLE_CREDENTIALS_PATH", 
    "./gen-lang-client-0974464525-ba886ad7f95a.json"
)

@dataclass
class GuidelineInfo:
    """
    Information about a guideline document.
    
    Attributes:
        id (str): Unique identifier for the guideline
        name (str): Name of the guideline document
        content (str): Content of the guideline document
        last_updated (str): Last update timestamp
        type (str): Type of guideline (e.g., "review", "amendment", "general")
    """
    id: str
    name: str
    content: str
    last_updated: str
    type: str

class GoogleDriveService:
    """
    Service class for Google Drive operations.
    
    This class handles all interactions with Google Drive API, including:
    - Authentication and service initialization
    - File listing and searching
    - File content retrieval
    - Guidelines management
    
    Attributes:
        service: Google Drive API service instance
        guidelines (Dict[str, GuidelineInfo]): Dictionary of loaded guidelines
    """
    
    def __init__(self, credentials_json: Optional[str] = None) -> None:
        """
        Initialize Google Drive service with credentials.
        
        Args:
            credentials_json (Optional[str]): Path to credentials JSON file or JSON string.
                If None, uses CREDENTIALS_PATH from environment.
        """
        self.service = self._initialize_service(credentials_json or CREDENTIALS_PATH)
        self.guidelines: Dict[str, GuidelineInfo] = {}
    
    def _initialize_service(self, credentials_json: str) -> Resource:
        """
        Initialize Google Drive service with credentials.
        
        Args:
            credentials_json (str): Path to credentials JSON file or JSON string
            
        Returns:
            googleapiclient.discovery.Resource: Initialized Google Drive service
            
        Raises:
            ValueError: If no credentials provided
            FileNotFoundError: If credentials file not found
            json.JSONDecodeError: If credentials file is invalid JSON
            Exception: If service initialization fails
        """
        try:
            if not credentials_json:
                raise ValueError("No credentials provided")
            
            # Load credentials from file
            with open(credentials_json, 'r') as f:
                credentials_info = json.load(f)
            
            credentials = ServiceAccountCredentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            
            return build('drive', 'v3', credentials=credentials)
            
        except FileNotFoundError:
            logger.error(f"Credentials file not found at {credentials_json}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in credentials file at {credentials_json}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            raise

    def load_guidelines(self) -> str:
        """
        Load and process guideline documents from Google Drive.
        
        Returns:
            str: Status message about loaded guidelines
        """
        try:
            # Search for guideline documents
            search_query = (
                "name contains 'Guideline' or "
                "name contains 'NDA' or "
                "name contains 'PROCUREMENT' or "
                "name contains 'Rules' or "
                "name contains 'Contract' or "
                "name contains 'Act' or "
                "name contains 'Reviewing' or "
                "name contains 'Amendment' or "
                "name contains 'Original'"
            )
            
            results = self.service.files().list(
                q=search_query,
                pageSize=50,
                fields="files(id,name,mimeType,modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            
            if not files:
                return "No guideline documents found in Google Drive."
            
            loaded_count = 0
            for file in files:
                try:
                    # Get file content
                    content = self.get_file_content(file['id'])
                    
                    # Create guideline info
                    guideline = GuidelineInfo(
                        id=file['id'],
                        name=file['name'],
                        content=content,
                        last_updated=file.get('modifiedTime', ''),
                        type=self._determine_guideline_type(file['name'])
                    )
                    
                    # Store in guidelines dictionary
                    self.guidelines[file['id']] = guideline
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing guideline file {file['name']}: {e}")
                    continue
            
            return f"Successfully loaded {loaded_count} guideline documents."
            
        except Exception as e:
            logger.error(f"Error loading guidelines: {e}")
            return f"Error loading guidelines: {str(e)}"

    def _determine_guideline_type(self, filename: str) -> str:
        """
        Determine the type of guideline based on filename.
        
        Args:
            filename (str): Name of the file
            
        Returns:
            str: Type of guideline ('review', 'amendment', or 'general')
        """
        filename_lower = filename.lower()
        if 'review' in filename_lower:
            return 'review'
        elif 'amendment' in filename_lower:
            return 'amendment'
        return 'general'

    def get_guideline_content(self, guideline_type: Optional[str] = None) -> str:
        """
        Get content of guidelines, optionally filtered by type.
        
        Args:
            guideline_type (Optional[str]): Type of guidelines to retrieve
            
        Returns:
            str: Formatted guideline content
        """
        if not self.guidelines:
            return "No guidelines loaded. Please use 'load guidelines' command first."
        
        response = "Available Guidelines:\n\n"
        
        for guideline in self.guidelines.values():
            if guideline_type and guideline.type != guideline_type:
                continue
                
            response += (
                f"{guideline.name}\n"
                f"   Last Updated: {guideline.last_updated[:10]}\n"
                f"   Type: {guideline.type}\n"
                f"   Content Preview: {guideline.content[:200]}...\n\n"
            )
        
        return response

    def list_nda_files(self, folder_id: Optional[str] = None) -> str:
        """
        List all NDA-related files from Google Drive.
        
        Returns:
            str: Formatted string containing list of NDA files with details
        """
        try:
            # Search for NDA documents with various keywords
            search_query = (
                "name contains 'NDA' or "
                "name contains 'Non-Disclosure' or "
                "name contains 'confidential'"
            )
            
            # Add folder specific search if folder_id is provided
            if folder_id:
                search_query = f"{search_query} and '{folder_id}' in parents"
            
            results = self.service.files().list(
                q=search_query,
                pageSize=50,
                fields="files(id,name,mimeType,modifiedTime,size,owners,webViewLink)"
            ).execute()
            
            files = results.get('files', [])
            
            if not files:
                return "No NDA documents found in Google Drive."
            
            response = "Found NDA documents in Google Drive:\n\n"
            for i, file in enumerate(files, 1):
                response += (
                    f"{i}. {file['name']}\n"
                    f"   Modified: {file.get('modifiedTime', 'Unknown')[:10]}\n"
                    f"   Link: {file.get('webViewLink', 'No link available')}\n"
                    f"   Type: {file.get('mimeType', 'Unknown')}\n"
                    f"   Size: {self._format_size(int(file.get('size', 0)))}\n\n"
                )
            
            response += "\nTo view a file's content, use the command: 'view file [number]'"
            return response
            
        except Exception as e:
            logger.error(f"Error listing NDA files: {e}")
            return f"Error accessing Google Drive: {str(e)}"

    def _format_size(self, size_bytes: int) -> str:
        """
        Format file size in bytes to human readable format.
        
        Args:
            size_bytes (int): Size in bytes
            
        Returns:
            str: Formatted size string (e.g., "1.5 MB")
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ("B", "KB", "MB", "GB", "TB")
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"

    def get_file_content(self, file_id: str) -> str:
        """
        Get content of a file from Google Drive.
        
        Args:
            file_id (str): ID of the file to retrieve
            
        Returns:
            str: File content or error message
            
        Raises:
            Exception: If file content cannot be retrieved
        """
        try:
            # Get file metadata
            file_metadata = self.service.files().get(fileId=file_id).execute()
            mime_type = file_metadata.get('mimeType', '')
            
            if mime_type == 'application/pdf':
                return self._get_pdf_content(file_id)
            elif 'google-apps' in mime_type:
                return self._get_google_apps_content(file_id, mime_type)
            else:
                return self._get_other_content(file_id)
                
        except Exception as e:
            logger.error(f"Error getting file content: {e}")
            return f"[Error reading file: {str(e)}]"

    def _get_pdf_content(self, file_id: str) -> str:
        """
        Get content from a PDF file.
        
        Args:
            file_id (str): ID of the PDF file
            
        Returns:
            str: Extracted text from PDF
        """
        request = self.service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            _, done = downloader.next_chunk()
            
        fh.seek(0)
        reader = PyPDF2.PdfReader(fh)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() or ""
            
        return text if text else "[No extractable text in PDF]"

    def _get_google_apps_content(self, file_id: str, mime_type: str) -> str:
        """
        Get content from a Google Apps file (Docs, Sheets, etc.).
        
        Args:
            file_id (str): ID of the Google Apps file
            mime_type (str): MIME type of the file
            
        Returns:
            str: Exported content from Google Apps file
        """
        export_mime_type = 'text/plain' if 'document' in mime_type else 'text/csv'
        content = self.service.files().export(
            fileId=file_id,
            mimeType=export_mime_type
        ).execute()
        return content.decode('utf-8', errors='ignore')

    def _get_other_content(self, file_id: str) -> str:
        """
        Get content from other file types.
        
        Args:
            file_id (str): ID of the file
            
        Returns:
            str: Decoded content from file
        """
        content = self.service.files().get_media(fileId=file_id).execute()
        return content.decode('utf-8', errors='ignore')

    def view_file(self, file_number: int) -> str:
        """
        View content of a specific file by its number in the list.
        
        Args:
            file_number (int): Number of the file in the list
            
        Returns:
            str: File content or error message
        """
        try:
            # Get list of files
            results = self.service.files().list(
                q=(
                    "name contains 'NDA' or "
                    "name contains 'Non-Disclosure' or "
                    "name contains 'confidential'"
                ),
                pageSize=50,
                fields="files(id,name)"
            ).execute()
            
            files = results.get('files', [])
            
            if not files:
                return "No NDA documents found in Google Drive."
            
            if file_number < 1 or file_number > len(files):
                return f"Invalid file number. Please choose a number between 1 and {len(files)}."
            
            # Get selected file
            selected_file = files[file_number - 1]
            file_id = selected_file['id']
            file_name = selected_file['name']
            
            # Get file content
            content = self.get_file_content(file_id)
            
            response = (
                f"Content of file: {file_name}\n\n"
                f"--- Content Start ---\n"
                f"{content}\n"
                f"--- Content End ---\n"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error viewing file: {e}")
            return f"Error viewing file: {str(e)}"

    def get_analysis_context(self, nda_content: str) -> str:
        """
        Prepares the context for NDA analysis including the NDA content and loaded guidelines.
        
        Args:
            nda_content (str): The content of the NDA document to analyze
            
        Returns:
            str: Formatted context string containing NDA content and guidelines
        """
        context_string = f"NDA Document Content:\n---\n{nda_content}\n---\n\n"

        if self.guidelines:
            context_string += "Loaded Guidelines:\n---\n"
            for name, guideline in self.guidelines.items():
                context_string += (
                    f"Guideline: {guideline.name} (Type: {guideline.type})\n"
                    f"{guideline.content}\n---\n"
                )
            context_string += "\n"
        else:
            context_string += "No specific guidelines loaded. Using general analysis criteria.\n\n"

        # Add static analysis instructions
        context_string += """
ANALYSIS INSTRUCTIONS FOR LLM:
You are an expert in NDA analysis. Analyze the 'NDA Document Content' provided above based on the 'Loaded Guidelines' (if any) and the following general criteria. Prioritize the Loaded Guidelines if they provide more specific instructions than the general criteria.

Follow the ANALYSIS OUTPUT FORMAT strictly.

General Analysis Criteria:
1. Check for essential NDA clauses:
   - Definition of Confidential Information
   - Obligations of Receiving Party
   - Permitted Use
   - Return of Information
   - Term/Duration
   - Remedies
   - Governing Law
   - Signatures

2. Look for potential issues:
   - Overly broad definitions
   - Unreasonable obligations
   - Unclear language
   - Missing essential protections

3. Provide specific recommendations:
   - Clear language suggestions
   - Missing clause templates
   - Compliance improvements

4. When analyzing, provide:
   - A summary of key findings
   - Missing or problematic clauses
   - Specific recommendations
   - Best practices
"""
        return context_string

# Create Google Drive Service instance
drive_service = GoogleDriveService()

# Enhanced Main Agent
enhanced_nda_agent = LlmAgent(
    name="enhanced_nda_assistant",
    model=MODEL_NAME,
    description="Enhanced NDA Document Analysis Assistant",
    instruction="""When starting a conversation, introduce yourself as your Enhanced NDA Assistant like this:
Hello! I am your Enhanced NDA Assistant. I am here to help you analyze Non-Disclosure Agreement documents efficiently.

Here's what I can do:
- **Process Files**: Analyze NDA files that you upload directly.
- **Google Drive Integration**: Connect to your Google Drive to list and view NDA documents.
- **Guidelines Integration**: Learn from specific guideline documents you provide to enhance my analysis.
- **Intelligent Analysis**: Provide detailed analysis of your NDA documents based on standard criteria and your guidelines.
- **Comprehensive Reporting**: Deliver structured reports on violations, unclear clauses, and suggested corrections.

My goal is to provide you with actionable insights to review and amend your NDA documents.

CAPABILITIES:
1. **Process Files**: You can directly process files uploaded by users. The files will be automatically available in the conversation.
2. **Google Drive Integration**: Access and analyze NDA documents from Google Drive.
3. **Guidelines Integration**: Learn from and apply guidelines for reviewing and amending documents.
4. **Intelligent Analysis**: Analyze NDA documents and provide detailed feedback.
5. **Comprehensive Reporting**: Provide detailed analysis with recommendations.

GOAL: Assist the user in analyzing NDA documents and providing actionable insights, always leveraging the loaded guidelines when performing analysis.

IMPORTANT WORKFLOW:
When a user wants to analyze documents:
1. Ask if they want to analyze local files or files from Google Drive.
2. For Google Drive files:
   - Use 'list_nda_files' to show available NDA documents.
   - Let user select which document to analyze using 'view_file'.
   - After using 'view_file', present the content received from the tool to the user in the chat.
   - User provides content or confirms file from 'view_file' for analysis. **Before calling the analysis tool, check if guidelines are loaded. If guidelines are loaded, explicitly state that you will now analyze the document using the provided content and the loaded guidelines, then proceed to use 'get_analysis_context'. If no guidelines are loaded, explicitly inform the user that only standard criteria will be used and ASK IF THEY WANT TO LOAD GUIDELINES using the 'Load guidelines' command BEFORE PROCEEDING with the analysis. Wait for their response. If they confirm to proceed without guidelines or don't respond about guidelines, then use 'get_analysis_context' with the document content.** Use the output from 'get_analysis_context' and the ANALYSIS OUTPUT FORMAT instructions below to perform the analysis and present the results.
3. For local files:
   - User provides content or uploads a file for analysis. **Before calling the analysis tool, check if guidelines are loaded. If guidelines are loaded, explicitly state that you will now analyze the document using the provided content and the loaded guidelines, then proceed to use 'get_analysis_context'. If no guidelines are loaded, explicitly inform the user that only standard criteria will be used and ASK IF THEY WANT TO LOAD GUIDELINES using the 'Load guidelines' command BEFORE PROCEEDING with the analysis. Wait for their response. If they confirm to proceed without guidelines or don't respond about guidelines, then use 'get_analysis_context' tool with the document content.** Use the output from 'get_analysis_context' and the ANALYSIS OUTPUT FORMAT instructions below to perform the analysis and present the results.
   - Process uploaded files directly.
4. Apply relevant guidelines for analysis:
   - Use 'load_guidelines' to load guideline documents.
   - Use 'get_guideline_content' to access guideline content (primarily for preview). The loaded guidelines will be automatically included in the context provided by the 'get_analysis_context' tool when you analyze a document.

ANALYSIS OUTPUT FORMAT:
Organize your output into the following sections:

1. 🚫 Violations Identified
   - List all paragraphs that violate the provided rules.
   - Use bullet points for each violation.
   - Clearly label:
     * Original NDA text
     * Explanation of violation
     * Reference to the rule: **Refer to the specific guideline document used (e.g., 'Guideline: [Guideline Name] (Type: [Guideline Type])' ) and if possible, the relevant section or rule number within that guideline.**
   - **Use bold text for key terms or phrases in the violation.**

2. 🤔 Unclear or Unreasonable Clauses
   - List any clauses that are ambiguous, contradictory, or illogical.
   - Use bullet points for each clause.
   - Clearly label:
     * Original text
     * Explanation of why it's problematic
     * Optional suggestion or rewrite.
   - **Use bold text for key terms or phrases in the problematic clause.**

3. ✅ Corrected Clauses
   - Provide rewritten versions of all problematic clauses.
   - Use bullet points for each corrected clause.
   - Format:
     * Original Clause: [Insert original paragraph]
     * Rewritten Clause: [Insert corrected version based on rules or clarity]
   - **Use bold text to highlight changes in the rewritten clause.**

4. 📊 Summary
   - Total rule violations found.
   - Total unclear/unreasonable clauses found.
   - Recommendation on whether the NDA is usable after correction.
   - Provide a clear overall assessment.

COMMANDS YOU UNDERSTAND:
- "List NDA files" - Use list_nda_files to show available NDA documents.
- "View file [number]" - Use view_file to display content of a specific file. After calling, present the content from the tool to the user.
- "Load guidelines" - Use load_guidelines to load guideline documents.
- "Show guidelines" - Use get_guideline_content to display available guidelines.
- "Analyze this document" - **Use 'get_analysis_context' tool with the document content**, then perform analysis based on the tool's output and the ANALYSIS OUTPUT FORMAT. You will need the document content for this command (either from a viewed file or uploaded).
- "What clauses are missing?" - Check for missing essential clauses. This should be part of the full analysis when you use 'Analyze this document'.
- "Is this NDA compliant?" - Check compliance with standard practices. This should be part of the full analysis when you use 'Analyze this document'.
- "How can this be improved?" - Provide specific recommendations. This should be part of the full analysis when you use 'Analyze this document'.
- "Exit" - End session.

ANALYSIS GUIDELINES:
Apply the following criteria and any loaded guideline documents when analyzing an NDA. Prioritize loaded guidelines if conflicts arise.

1. Check for essential NDA clauses:
   - Definition of Confidential Information
   - Obligations of Receiving Party
   - Permitted Use
   - Return of Information
   - Term/Duration
   - Remedies
   - Governing Law
   - Signatures

2. Look for potential issues:
   - Overly broad definitions
   - Unreasonable obligations
   - Unclear language
   - Missing essential protections
""",
    tools=[
        drive_service.list_nda_files,
        drive_service.view_file,
        drive_service.load_guidelines,
        drive_service.get_guideline_content,
        drive_service.get_analysis_context
    ]
)

# Set as root agent
root_agent = enhanced_nda_agent