# Standard library imports
import os
import json
import logging
import base64
import openpyxl
from datetime import datetime
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Any, Union

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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration constants
MODEL_NAME: str = os.getenv("MODEL", "gemini-2.0-flash")
CREDENTIALS_PATH: str = os.getenv(
    "GOOGLE_CREDENTIALS_PATH", "./gen-lang-client-0974464525-ba886ad7f95a.json"
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
            with open(credentials_json, "r") as f:
                credentials_info = json.load(f)

            credentials = ServiceAccountCredentials.from_service_account_info(
                credentials_info,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )

            return build("drive", "v3", credentials=credentials)

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

            results = (
                self.service.files()
                .list(
                    q=search_query,
                    pageSize=50,
                    fields="files(id,name,mimeType,modifiedTime)",
                )
                .execute()
            )

            files = results.get("files", [])

            if not files:
                return "No guideline documents found in Google Drive."

            loaded_count = 0
            for file in files:
                try:
                    # Get file content
                    content = self.get_file_content(file["id"])

                    # Create guideline info
                    guideline = GuidelineInfo(
                        id=file["id"],
                        name=file["name"],
                        content=content,
                        last_updated=file.get("modifiedTime", ""),
                        type=self._determine_guideline_type(file["name"]),
                    )

                    # Store in guidelines dictionary
                    self.guidelines[file["id"]] = guideline
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
        if "review" in filename_lower:
            return "review"
        elif "amendment" in filename_lower:
            return "amendment"
        return "general"

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

            results = (
                self.service.files()
                .list(
                    q=search_query,
                    pageSize=50,
                    fields="files(id,name,mimeType,modifiedTime,size,owners,webViewLink)",
                )
                .execute()
            )

            files = results.get("files", [])

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

            response += (
                "\nTo view a file's content, use the command: 'view file [number]'"
            )
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
            mime_type = file_metadata.get("mimeType", "")

            if mime_type == "application/pdf":
                return self._get_pdf_content(file_id)
            elif "google-apps" in mime_type:
                return self._get_google_apps_content(file_id, mime_type)
            elif mime_type in [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
                "application/vnd.ms-excel",  # .xls
            ]:
                return self._get_excel_content_via_openpyxl(file_id)
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
        text = ""
        try:
            # Try extracting text using PyPDF2
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                text += page.extract_text() or ""

            # If standard text extraction is minimal or failed, attempt OCR
            if not text or len(text.strip()) < 50:
                logger.info(
                    f"Minimal text extracted from {file_id} with PyPDF2, attempting OCR."
                )
                fh.seek(0)  # Reset stream position to read for OCR
                # --- OCR Implementation Needed Here ---
                # This section requires an OCR library (e.g., pytesseract) and potentially a PDF-to-image converter (e.g., pdf2image).
                # Example (conceptual, requires libraries and setup):
                # try:
                #     from PIL import Image
                #     import pytesseract
                #     from pdf2image import convert_from_bytes
                #
                #     images = convert_from_bytes(fh.read())
                #     ocr_text = ""
                #     for i, image in enumerate(images):
                #         ocr_text += pytesseract.image_to_string(image, lang='tha+eng') # Specify languages if needed
                #         ocr_text += "\n" # Add newline between pages
                #     text = ocr_text if ocr_text.strip() else "[OCR could not extract text]"
                #     if text.strip() == "[OCR could not extract text]":
                #          logger.warning(f"OCR failed to extract text for {file_id}.")
                #     else:
                #          logger.info(f"OCR successfully extracted text for {file_id}.")
                #
                # except ImportError:
                #     logger.error("Required libraries for OCR (e.g., pytesseract, pdf2image, PIL) not installed.")
                #     text = "[Error: Required libraries for OCR not installed.]"
                # except Exception as ocr_e:
                #     logger.error(f"Error during OCR processing for {file_id}: {ocr_e}")
                #     text = f"[Error during OCR: {ocr_e}]"

                # Placeholder text if OCR is not implemented or fails
                if (
                    not text
                    or text.strip() == "[OCR could not extract text]"
                    or text.startswith("[Error")
                ):  # Check if OCR attempt was successful or implemented
                    text = "[Content could not be extracted. It might be an image-based PDF without selectable text, and OCR is not fully implemented or failed.]"

        except Exception as e:
            logger.error(f"Error getting PDF content for {file_id} with PyPDF2: {e}")
        return text if text else "[No extractable text in PDF]"

    def _get_google_apps_content(self, file_id: str, mime_type: str) -> str:
        """
        Get content from a Google Apps file (Docs, Sheets, etc.) or exported Excel file.

        Args:
            file_id (str): ID of the Google Apps or Excel file
            mime_type (str): MIME type of the file

        Returns:
            str: Exported content from Google Apps or Excel file
        """
        if "document" in mime_type:
            export_mime_type = "text/plain"
        elif "spreadsheet" in mime_type or mime_type in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
            "application/vnd.ms-excel",  # .xls
        ]:
            export_mime_type = "text/csv"
        else:
            # Default or handle other Google Apps types if necessary, though text/plain is common
            export_mime_type = "text/plain"

        content = (
            self.service.files()
            .export(fileId=file_id, mimeType=export_mime_type)
            .execute()
        )
        return content.decode("utf-8", errors="ignore")

    def _get_excel_content_via_openpyxl(self, file_id: str) -> Dict[str, Any]:
        """
        Gets Excel content by downloading the binary file and reading with openpyxl.

        Args:
            file_id (str): ID of the Excel file.

        Returns:
            Dict[str, Any]: Structured Excel data.
        """
        try:
            # Download the raw Excel file
            request = self.service.files().get_media(fileId=file_id)
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            fh.seek(0)

            # Load the workbook with openpyxl
            # data_only=True retrieves cell values, not formulas
            wb = openpyxl.load_workbook(fh, data_only=True)
            file_metadata = self.service.files().get(fileId=file_id, fields="name").execute()
            file_name = file_metadata.get('name', 'Unknown')

            result = {
                'file_name': file_name,
                'file_id': file_id,
                'sheets': {},
                'summary': f"Excel file '{file_name}' has {len(wb.sheetnames)} sheets: {', '.join(wb.sheetnames)}",
            }

            # Read sample data from each sheet (e.g., first 5 rows, first 5 columns)
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows_data = []
                # Adjust max_row and max_col as needed to get more sample data
                max_data_row = min(ws.max_row, 5)
                max_data_col = min(ws.max_column, 5)
                
                # Read headers if the first row seems like headers
                headers = []
                if max_data_row >= 1:
                    header_row = [ws.cell(row=1, column=c).value for c in range(1, max_data_col + 1)]
                    # Simple check if first row looks like headers (e.g., not all empty, contains some text)
                    if any(cell is not None and str(cell).strip() for cell in header_row):
                         headers = header_row
                         start_row = 2 # Start data from the second row
                    else:
                         start_row = 1 # Start data from the first row (no header row)

                    # Read sample data rows (excluding header if identified)
                    for r in range(start_row, max_data_row + 1):
                         row_values = [ws.cell(row=r, column=c).value for c in range(1, max_data_col + 1)]
                         rows_data.append(row_values)


                result['sheets'][sheet_name] = {
                    'headers': headers,
                    'sample_data': rows_data,
                    'row_count': ws.max_row,
                    'col_count': ws.max_column
                }

            return result

        except Exception as e:
            logger.error(f"Error reading Excel with openpyxl for {file_id}: {e}")
            return {'error': str(e), 'summary': f'Error reading Excel file: {str(e)}'}

    def _get_other_content(self, file_id: str) -> str:
        """
        Get content from other file types.
        
        Args:
            file_id (str): ID of the file
            
        Returns:
            str: Decoded content from file
        """
        content = self.service.files().get_media(fileId=file_id).execute()
        return content.decode("utf-8", errors="ignore")

    def _get_excel_with_openpyxl(self, file_id: str) -> Dict[str, Any]:
        try:
            # ดาวน์โหลดไฟล์ Excel จาก Google Drive
            request = self.service.files().get_media(fileId=file_id)
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)

            # โหลดด้วย openpyxl
            wb = openpyxl.load_workbook(fh, data_only=True)
            result = {
                "sheets": {},
                "file_id": file_id,
                "summary": f"Excel file has {len(wb.sheetnames)} sheets: {', '.join(wb.sheetnames)}",
            }

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows = []
                for row in ws.iter_rows(min_row=1, max_row=5, min_col=1, max_col=2, values_only=True):
                    rows.append(row)
                result["sheets"][sheet_name] = rows

            return result

        except Exception as e:
            logger.error(f"openpyxl error for {file_id}: {e}")
            return {"error": str(e)}

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
            results = (
                self.service.files()
                .list(
                    q=(
                        "name contains 'NDA' or "
                        "name contains 'Non-Disclosure' or "
                        "name contains 'confidential'"
                    ),
                    pageSize=50,
                    fields="files(id,name)",
                )
                .execute()
            )

            files = results.get("files", [])

            if not files:
                return "No NDA documents found in Google Drive."

            if file_number < 1 or file_number > len(files):
                return f"Invalid file number. Please choose a number between 1 and {len(files)}."

            # Get selected file
            selected_file = files[file_number - 1]
            file_id = selected_file["id"]
            file_name = selected_file["name"]

            # Get file content
            content = self.get_file_content(file_id)

            # Check if content is a dictionary (likely from Excel processed by openpyxl)
            if isinstance(content, dict):
                # Format the dictionary content for display
                file_name = content.get("file_name", "Unknown Excel File")
                summary = content.get("summary", "Could not retrieve summary.")
                sheets_data = content.get("sheets", {})

                formatted_content = f"File: {file_name}\n\n{summary}\n\n"

                # Limit the number of sheets shown for brevity
                for sheet_name, sheet_data in list(sheets_data.items())[:3]:
                    formatted_content += f"  --- Sheet: {sheet_name} ---\n"

                    # Check for headers and sample_data keys
                    if 'headers' in sheet_data and sheet_data['headers']:
                        formatted_content += "  Headers: " + ", ".join(map(str, sheet_data['headers'])) + "\n"

                    if 'sample_data' in sheet_data and sheet_data['sample_data']:
                        formatted_content += "  Sample Rows:\n"
                        for i, row_data in enumerate(sheet_data['sample_data']):
                            row_string = ", ".join(map(lambda x: str(x) if x is not None else "", row_data))
                            formatted_content += f"    Row {i + 1}: {row_string}\n"
                    else:
                        formatted_content += "  [Sheet is empty or no sample data available]\n"

                    if 'error' in content and content['error']:
                        formatted_content += f"  Error in data: {content['error']}\n"

                    formatted_content += "\n"  # Add spacing between sheets

                # Return only the formatted content string for Excel, without the outer markdown block
                return formatted_content

            else:
                # Original formatting for string content (for non-Excel files)
                formatted_content = "".join(
                    [line.strip() + "\n" for line in content.splitlines()]
                )
                formatted_content = formatted_content.replace(
                    "\n\n\n", "\n\n"
                )  # Reduce multiple newlines to double newlines

                response = (
                    f"Content of file: {file_name}\n\n" f"```\n{formatted_content}\n```\n"
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
            context_string += (
                "No specific guidelines loaded. Using general analysis criteria.\n\n"
            )

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

    def analyze_excel_structure(self, file_id: str) -> Dict[str, Any]:
        """
        Analyze Excel file structure in detail.

        Args:
            file_id (str): ID of the Excel file

        Returns:
            Dict[str, Any]: Detailed analysis of Excel structure
        """
        try:
            excel_data = self._get_excel_content_enhanced(file_id, "")

            if excel_data.get("error"):
                return {"error": excel_data["error"]}

            analysis = {
                "file_info": {
                    "name": excel_data["file_name"],
                    "size": excel_data.get("file_size", 0),
                    "mime_type": excel_data["mime_type"],
                },
                "structure": {},
                "data_types": {},
                "recommendations": [],
            }

            # Analyze CSV data if available
            if excel_data.get("raw_csv"):
                lines = excel_data["raw_csv"].split("\n")
                if len(lines) > 1:
                    headers = [col.strip('"').strip() for col in lines[0].split(",")]

                    # Analyze data types in each column
                    sample_rows = lines[
                        1 : min(6, len(lines))
                    ]  # Sample first 5 data rows

                    for i, header in enumerate(headers):
                        column_samples = []
                        for row in sample_rows:
                            cols = row.split(",")
                            if i < len(cols):
                                column_samples.append(cols[i].strip('"').strip())

                        # Simple data type detection
                        data_type = self._detect_column_type(column_samples)
                        analysis["data_types"][header] = {
                            "detected_type": data_type,
                            "samples": column_samples[:3],  # First 3 samples
                        }

                    analysis["structure"] = {
                        "total_rows": len(lines) - 1,
                        "total_columns": len(headers),
                        "headers": headers,
                        "has_headers": True,
                    }

                    # Generate recommendations
                    if len(headers) > 20:
                        analysis["recommendations"].append(
                            "Large number of columns - consider data normalization"
                        )

                    if len(lines) > 1000:
                        analysis["recommendations"].append(
                            "Large dataset - consider processing in chunks"
                        )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing Excel structure: {e}")
            return {"error": str(e)}

    def _detect_column_type(self, samples: List[str]) -> str:
        """
        Simple data type detection for Excel columns.

        Args:
            samples (List[str]): Sample values from the column

        Returns:
            str: Detected data type
        """
        if not samples or all(not sample.strip() for sample in samples):
            return "empty"

        # Remove empty samples
        non_empty_samples = [s.strip() for s in samples if s.strip()]

        if not non_empty_samples:
            return "empty"

        # Check for numbers
        numeric_count = 0
        date_count = 0

        for sample in non_empty_samples:
            # Check if numeric
            try:
                float(sample.replace(",", ""))  # Handle comma-separated numbers
                numeric_count += 1
            except ValueError:
                pass

            # Check if date-like
            if any(sep in sample for sep in ["/", "-", "."]):
                if len(sample.split("/")) == 3 or len(sample.split("-")) == 3:
                    date_count += 1

        total_samples = len(non_empty_samples)

        if numeric_count == total_samples:
            return "numeric"
        elif date_count > total_samples * 0.5:
            return "date"
        elif numeric_count > total_samples * 0.5:
            return "mixed_numeric"
        else:
            return "text"


# Create Google Drive Service instance
drive_service = GoogleDriveService()

# --- Define Specialized Sub-Agents ---

# Agent for listing NDA files
nda_listing_agent = LlmAgent(
    name="nda_listing_agent",
    model=MODEL_NAME,
    description="Agent specializing in listing available NDA files from integrated storage like Google Drive.",
    instruction="""You are the NDA Listing Agent. Your SOLE purpose is to list available NDA files from Google Drive when explicitly instructed by the Main Agent or the user. You are a step in the workflow orchestrated by the Main Agent.

Follow these steps:

1.  Receive the delegation or command to list files.
2.  Use the `list_nda_files` tool provided to search for and retrieve the list of NDA documents.
3.  Present the results from the `list_nda_files` tool clearly to the user. Ensure the list is formatted for easy reading, showing file numbers and names.
4.  **Crucially, after presenting the list, your task is complete.** Do not attempt to interpret the user's next action or initiate further steps like viewing files or analysis. The Main Agent will receive the user's next instruction and determine the subsequent action based on their response (e.g., selecting a file number). **Signal completion** after presenting the list.

Available Tools:
- list_nda_files: Use this tool to get a list of NDA documents from Google Drive.

ERROR HANDLING: If the `list_nda_files` tool reports an error, inform the user clearly about the error and state that the listing could not be completed. Suggest they report this issue to the Main Assistant. **Signal completion** after reporting the error, returning control to the Main Agent.
""",
    tools=[drive_service.list_nda_files],
)

# Agent for viewing specific NDA file content
nda_viewing_agent = LlmAgent(
    name="nda_viewing_agent",
    model=MODEL_NAME,
    description="Agent specializing in viewing the content of a specific NDA file by file number from a list provided by the listing agent.",
    instruction="""You are the NDA Viewing Agent. Your SOLE purpose is to retrieve the content of a specific NDA file from Google Drive, identified by a file number provided by the user or the Main Agent. You are a step in the workflow orchestrated by the Main Agent. You will then provide a concise summary of the content instead of the full text, **UNLESS the file is a spreadsheet (like .xls or .xlsx)**.

Follow these steps precisely:

1.  Receive the delegation or command to view a file, along with the specified file number.
2.  Use the `view_file` tool provided with the file number. This tool will return the file content or an error.
3.  **Handle Tool Output**: Carefully examine the result from the `view_file` tool.
    a.  **If the tool reports an Error**: Inform the user clearly about the error message received from the tool and state that the file content could not be retrieved. Suggest they report this issue to the Main Assistant. **Signal completion** after reporting the error, returning control to the Main Agent.
    b.  **If the tool returns Content**: Examine the file name or associated mime type information (if available from the tool output or context, or infer from file extension like .xls or .xlsx). If it indicates a spreadsheet file, follow the instructions in step 4a. Otherwise, follow the instructions in step 4b.

4.  **Process Content Based on File Type**:
    a.  **If it's a Spreadsheet**: Recognize that the agent cannot summarize or analyze spreadsheet content directly using the language model. Output a clear message to the user stating that you have retrieved the content but cannot process it further for summarization or analysis. Present the raw text content received from the `view_file` tool (which should be in a readable format like CSV after recent updates to `view_file`) within a markdown code block (```csv\n...\n```) for readability. **Your task is complete after presenting this message and the raw content.** Do NOT attempt to summarize or delegate for analysis. **Signal completion** after presenting the content, returning control to the Main Agent.

    b.  **If it's NOT a Spreadsheet**: Proceed with summarizing the content. **Do NOT present the full file content directly to the user.** Based on the content received from the `view_file` tool (which should be plain text for non-spreadsheets), generate a concise summary (e.g., mention the type of document, key parties, approximate length, and date if available). Also, confirm that the full content has been successfully retrieved and is ready for analysis by the Analysis Agent.
5.  **Present Summary**: Present ONLY the summary and the confirmation message to the user. **Your task is complete after presenting the summary and confirmation.** Do not attempt to analyze the content or ask further questions about it yourself. The Main Agent or Analysis Agent will handle the next step based on user input. **Signal completion** after presenting the summary, returning control to the Main Agent.

Available Tools:
- view_file: Use this tool with a file number to get the content of a specific NDA document from Google Drive.

ERROR HANDLING: (Covered in step 3a) If the `view_file` tool reports an error (e.g., invalid file number, access issue), inform the user clearly about the error and suggest they report it to the Main Assistant. If the file number is out of the valid range, state the valid range to the user (this check might happen before the tool call or be part of the tool's error reporting - the Agent should handle the tool's error output). **Signal completion** after reporting the error.
""",
    tools=[drive_service.view_file],
)

# Agent for managing guideline documents
nda_guidelines_agent = LlmAgent(
    name="nda_guidelines_agent",
    model=MODEL_NAME,
    description="Agent specializing in loading and displaying guideline documents relevant to NDA analysis.",
    instruction="""You are the NDA Guidelines Agent. Your SOLE purpose is to load guideline documents or display their content when explicitly instructed by the Main Agent or the user. You are a step in the workflow orchestrated by the Main Agent.

Follow these steps precisely:

1.  Receive the delegation or command, which will be either 'Load guidelines' or 'Show guidelines'.
2.  **Execute Action**: Use the appropriate tool based on the command:
    a.  If the command is 'Load guidelines', use the `load_guidelines` tool.
    b.  If the command is 'Show guidelines', use the `get_guideline_content` tool.
3.  **Handle Tool Output**: Carefully examine the result from the tool.
    a.  **If the tool reports an Error**: Inform the user clearly about the error message received from the tool and state that the action could not be completed. Suggest they report this issue to the Main Assistant. **Signal completion** after reporting the error, returning control to the Main Agent.
    b.  **If the tool returns Results**: Present the results (status message for loading, formatted content for showing) clearly to the user.
4.  **Signal Completion**: After performing the requested action and presenting the result (or error), your task is complete. Do not attempt to interpret the user's next action or initiate analysis or other tasks. **Signal completion** after presenting the results, returning control to the Main Agent.

Available Tools:
- load_guidelines: Use this tool to fetch and load guideline documents.
- get_guideline_content: Use this tool to retrieve and display the content of currently loaded guidelines.

ERROR HANDLING: (Covered in step 3a) If either tool reports an error, inform the user clearly about the error and suggest they report it to the Main Assistant. **Signal completion** after reporting the error.
""",
    tools=[drive_service.load_guidelines, drive_service.get_guideline_content],
)

# Agent for comprehensive NDA analysis
nda_analysis_agent = LlmAgent(
    name="nda_analysis_agent",
    model=MODEL_NAME,
    description="Specialized agent for comprehensive analysis and review of Non-Disclosure Agreement documents based on provided content and loaded guidelines, following a specific step-by-step workflow.",
    instruction="""You are the NDA Analysis Support Agent. Your purpose is to perform detailed, expert analysis of NDA documents. You receive control from the Main Agent ONLY when the user's request is clearly about analyzing, reviewing, or checking an NDA document, AND the document content has already been provided or identified and passed to you.

Your tasks, performed sequentially after receiving the document content, are:

1.  **Acknowledge Handover**: Start by acknowledging that you have taken over for NDA analysis and that you have received the document content. For example: "Okay, I have the document content now and will begin the analysis process."
2.  **Ask Legal System**: **IMMEDIATELY after acknowledging handover, ASK THE USER TO SPECIFY THE LEGAL SYSTEM FOR COMPARISON: "Is the legal system for comparison Thailand or Singapore?". Wait for their explicit response.** Do NOT proceed until the user provides the legal system.
3.  **Address Guidelines**: Once the user specifies the legal system, inform them about the guideline status before proceeding. You don't have a direct tool to check the internal state of the GoogleDriveService from here, but you should assume that if the user previously used the 'Load guidelines' command (handled by the Guidelines Agent), the guidelines are available to `get_analysis_context`. Based on whether you expect guidelines to be loaded, inform the user:
    -   If guidelines are expected to be loaded: "Okay, I will now analyze the document using the provided content, focusing on [Specified Legal System], and applying the loaded guidelines."
    -   If no guidelines are expected to be loaded (or if unsure, default to this): "Okay, I will now analyze the document using the provided content, focusing on [Specified Legal System]. No specific guidelines are currently loaded, so I will use general criteria. Would you like to load guidelines using the 'Load guidelines' command *before* I proceed with the analysis?" **Wait for their response.** If they explicitly confirm to proceed without guidelines, or if they do not respond after a reasonable pause, you may proceed.
4.  **Prepare Context**: Use the `get_analysis_context` tool with the document content that was passed to you. This tool will combine the NDA content and any currently loaded guidelines into the input for the analysis.
5.  **Perform Analysis**: Based on the output from `get_analysis_context` and the detailed ANALYSIS OUTPUT FORMAT instructions below, perform the comprehensive analysis.
6.  **Present Results**: Present the analysis results clearly to the user, **STRICTLY following the ANALYSIS OUTPUT FORMAT and utilizing Markdown for excellent readability and visual appeal (headings, bullet points, bold text, etc.)**. Ensure sections are clearly delineated.
7.  **Handle Follow-up**: Be ready to answer follow-up questions specifically related to the analysis or suggested changes. If the user asks about something outside of the analysis scope (e.g., listing files, loading different guidelines), recognize this and indicate that you need to transfer the conversation back to the Main Agent.

Available Tools:
- get_analysis_context: Use this tool with the document content to prepare the full context for analysis (combines NDA content and loaded guidelines). You will call this tool *after* the user specifies the legal system and guideline status is addressed.

ANALYSIS OUTPUT FORMAT (STRICTLY ADHERE TO THIS MARKDOWN FORMATTING FOR READABILITY):

### 🚫 Violations Identified

- List each violation clearly using bullet points.
- For each violation, include:

  - **Original Text**: [Quote the problematic content]<br>

  - **Explanation**: [Explain why the text violates the rule, citing the specific guideline document and rule if possible]<br>

  - **Reference**: [Specify the guideline document, any relevant section/rule number, **Page and Paragraph number (if found) - aim to include both if possible, using the format (Page Y, Paragraph Z)**]<br>

---

### 🤔 Unclear or Unreasonable Clauses

- List each unclear or unreasonable clause using bullet points.
- For each clause, include:

  - **Original Text**: [Quote the problematic content]<br>

  - **Explanation**: [Explain why the text is ambiguous, contradictory, or illogical]<br>

  - **Suggestion**: [Optional: Provide a suggestion or rewrite for clarity]<br>

  - **Reference**: [Specify any relevant guideline document and section/rule number, **Page and Paragraph number (if found) - aim to include both if possible, using the format (Page Z, Paragraph A)**]<br>

---

### ✅ Corrected Clauses

- Provide rewritten versions for problematic clauses using bullet points.
- Format each entry as:

  - **Original Clause**: [Insert original paragraph]<br>

  - **Rewritten Clause**: [Insert corrected version based on rules or clarity, highlight changes using **bold text**]<br>

---

### 📊 Completeness Check

- Check for the presence of essential NDA clauses listed in the General Analysis Criteria (Definition of Confidential Information, Obligations of Receiving Party, Permitted Use, Return of Information, Term/Duration, Remedies, Governing Law, Signatures).
- Use bullet points (`- `) for each clause.
- Indicate whether the clause is **Present** or **Missing**. If Present, briefly mention where it is or its key aspect. If Missing, state that it is missing.

---

### 📑 Summary

- Provide a concise summary of the analysis using bullet points where appropriate.
- Include:

  **Total Violations**: [Number of rule violations found]

  **Total Unclear Clauses**: [Number of unclear/unreasonable clauses found]

  **Overall Assessment**: [Recommendation on whether the NDA is usable after corrections]

ERROR HANDLING: If the `get_analysis_context` tool reports an error, inform the user clearly about the error and explain that the analysis cannot proceed. If you encounter issues during the analysis process itself (e.g., content is unclear), inform the user and suggest they clarify or provide different content.
""",
    tools=[drive_service.get_analysis_context],
)

# Add new agent for Excel operations
excel_operations_agent = LlmAgent(
    name="excel_operations_agent",
    model=MODEL_NAME,
    description="Agent specializing in Excel file operations including detailed analysis and structure inspection.",
    instruction="""You are the Excel Operations Agent. Your purpose is to handle detailed Excel file operations including structure analysis and data inspection.

Available Commands:
1. "analyze excel [file_id]" - Perform detailed structure analysis of an Excel file
2. "inspect excel [file_number]" - Inspect Excel file from the listed files

Follow these steps:

1. When asked to analyze an Excel file:
   - Use the `analyze_excel_structure` tool with the provided file_id
   - Present the analysis results in a clear, structured format
   - Include file info, structure details, data types, and recommendations

2. When asked to inspect an Excel file by number:
   - First get the file list to identify the correct file_id
   - Then use the analysis tools to provide detailed information

Available Tools:
- analyze_excel_structure: Analyze Excel file structure and data types
- list_nda_files: Get list of files to identify file_id from file number

Present results using clear formatting with:
- File information summary
- Data structure overview  
- Column data types
- Sample data preview
- Recommendations for data processing

ERROR HANDLING: If any tool reports an error, inform the user clearly and suggest alternative approaches.
""",
    tools=[drive_service.analyze_excel_structure, drive_service.list_nda_files],
)

# Main Agent (Root Agent for handling initial interaction and delegating to specialized sub-agents)
enhanced_nda_agent = LlmAgent(
    name="enhanced_nda_assistant",
    model=MODEL_NAME,
    description="Main Assistant for handling user inquiries about NDAs and delegating specific tasks like listing, viewing, guidelines management, and analysis to specialized support agents.",
    instruction="""Hello! I am your Enhanced NDA Assistant, ready to assist you with Non-Disclosure Agreement documents 🤖📄. I can manage files from Google Drive, use guidelines for analysis, perform detailed analysis, and provide structured reports. Ready to assist you! 👍

CAPABILITIES:
- Google Drive Integration (List/View files)
- Guidelines Management (Load/Show guidelines)
- Intelligent NDA Analysis
- Comprehensive Reporting
- Excel File Structure Analysis

COMMANDS I UNDERSTAND:
- "Analyze this document" (Starts analysis workflow)
- "List NDA files"
- "View file [number]" (Context-dependent)
- "Load guidelines"
- "Show guidelines"
- "Analyze Excel [file_id]" or "Inspect Excel [file_number]" (Delegates to Excel Agent)
- General NDA questions
- "Exit"

Your role is to understand user requests and delegate to specialized sub-agents. You do NOT use tools directly. You are the orchestrator of the workflow.

IMPORTANT WORKFLOW for Document Analysis (Initiated by "Analyze this document"):
When a user asks to analyze a document, initiate this specific multi-step workflow:
1.  **Ask for Source**: Ask the user: "Is the document you want to analyze a local file or is it in Google Drive?"
2.  **Handle Google Drive Source**: If the user specifies "Google Drive":
    a.  **Delegate Listing**: Delegate the task to `nda_listing_agent` to show the user available files. State clearly to the user that you are listing the files.
    b.  **Prompt for File Number**: After the `nda_listing_agent` presents the list, wait for the user to provide the number of the file they wish to view or analyze.
    c.  **Delegate Viewing**: Once the user provides a file number, delegate the task to `nda_viewing_agent` with the specified file number. State clearly to the user that you are retrieving the content of the file. **The `nda_viewing_agent` is instructed to provide a summary (for non-spreadsheets) or raw content (for spreadsheets), not the full text.**
    d.  **Receive and Confirm Content**: The `nda_viewing_agent` will provide a summary/raw content and confirmation of successful content retrieval. Inform the user that the content has been retrieved (referencing the summary if provided) and is ready for analysis (for supported file types). For spreadsheets, state that the content is retrieved but direct analysis is not possible via LLM.
    e.  **Delegate Analysis**: After confirming content retrieval for a supported file type (non-spreadsheet), delegate the task to the `nda_analysis_agent`, passing the document content received from `nda_viewing_agent` as part of the delegation context. State clearly to the user that you are now transferring the analysis task to the specialized Analysis Agent. **Do NOT delegate spreadsheet content for analysis.**
3.  **Handle Local File Source**: If the user specifies "Local file" or pastes content directly:
    a.  **Confirm Content**: Confirm that you will proceed with the analysis using the provided content.
    b.  **Delegate Analysis**: Delegate the task to the `nda_analysis_agent`, passing the local document content as part of the delegation context. State clearly to the user that you are now transferring the analysis task to the specialized Analysis Agent.

Delegate other specific commands directly to the appropriate sub-agent. After delegating, **wait for the sub-agent to complete its task and for the user's next instruction.**
- "List NDA files" -> Delegate to `nda_listing_agent`. Inform the user you are listing files.
- "View file [number]" -> If the user provides a number WITHOUT first asking to list files or asking to analyze a Google Drive file, ask the user to first list the files or initiate the analysis workflow. If this command follows a listing or analysis request, delegate to `nda_viewing_agent`. Inform the user you are retrieving the file content.
- "Load guidelines" -> Delegate to `nda_guidelines_agent`. Inform the user you are loading guidelines.
- "Show guidelines" -> Delegate to `nda_guidelines_agent`. Inform the user you are showing loaded guidelines.
- "Analyze Excel [file_id]" or "Inspect Excel [file_number]" -> Delegate to `excel_operations_agent`. Inform the user you are performing Excel operations.

Handle general questions about NDAs directly using your knowledge base. For example, explain what an NDA is, common clauses, or the importance of review.

If the user's request does not fit one of the specific commands or initiate a known workflow, respond to them directly based on their query using your general knowledge.

ERROR HANDLING: If a delegated agent or a tool call within a delegated agent reports an error, the sub-agent is instructed to inform the user. As the Main Agent, if you receive an error event from a sub-agent, acknowledge it, inform the user that the specific task failed, and guide them on potential next steps, such as trying again or rephrasing the request. Always strive to bring the conversation back to a state where you can effectively assist based on supported capabilities.
""",
    tools=[],  # Main agent delegates tool use to sub-agents
    sub_agents=[
        nda_listing_agent,
        nda_viewing_agent,
        nda_guidelines_agent,
        nda_analysis_agent,
        excel_operations_agent,
    ],
)

# Set as root agent
root_agent = enhanced_nda_agent