# openai_agents_advanced.py
import os
import json
import pandas as pd
from openai import AzureOpenAI
import inspect
from dotenv import load_dotenv

load_dotenv()

class AzureOpenAIToolAgents:
    """Enhanced agents with tool/function calling capabilities"""
    
    def __init__(self):
        api_key = os.getenv('AZURE_OPENAI_KEY')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        model = os.getenv('AZURE_OPENAI_DEPLOYMENT')
        
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.model = model
        self.define_tools()
    
    def define_tools(self):
        """Define tools available to agents"""
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_column_content",
                    "description": "Analyze column content to determine if it should be in fact or dimension table",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column_name": {"type": "string", "description": "Column name"},
                            "unique_ratio": {"type": "number", "description": "Unique values / total rows"},
                            "data_samples": {"type": "array", "description": "Sample data values"}
                        },
                        "required": ["column_name", "unique_ratio", "data_samples"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "map_sql_datatype",
                    "description": "Map Python/detected datatype to SQL Server datatype",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column_name": {"type": "string"},
                            "detected_type": {"type": "string"},
                            "max_length": {"type": "integer", "description": "Max string length"},
                            "decimal_places": {"type": "integer", "description": "For numeric types"}
                        },
                        "required": ["column_name", "detected_type"]
                    }
                }
            }
        ]
    
    def analyze_csv_with_tools(self, df, csv_filename):
        """Analyze CSV using tool calls"""
        
        # Prepare data for analysis
        analysis_data = {
            "filename": csv_filename,
            "row_count": len(df),
            "columns": []
        }
        
        for col in df.columns:
            col_data = {
                "name": col,
                "detected_type": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
                "unique_ratio": round(df[col].nunique() / len(df), 3),
                "samples": df[col].astype(str).head(5).tolist()
            }
            analysis_data["columns"].append(col_data)
        
        prompt = f"""
        Analyze this CSV data structure and determine the best fact/dimension split:
        
        {json.dumps(analysis_data, indent=2)}
        
        For each column, use the analyze_column_content tool to determine if it belongs to:
        1. FACT table (transactional, measures, metrics)
        2. DIMENSION tables (descriptive, attributes)
        
        Then provide your final recommendation.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data warehouse architect. Use provided tools to analyze column placement."},
                {"role": "user", "content": prompt}
            ],
            tools=self.tools,
            tool_choice="auto"
        )
        
        return self._process_tool_response(response, analysis_data)
    
    def detect_datatypes_with_tools(self, df):
        """Detect datatypes using tool calls"""
        
        column_info = {}
        for col in df.columns:
            column_info[col] = {
                "detected_type": str(df[col].dtype),
                "max_length": df[col].astype(str).str.len().max() if df[col].dtype == 'object' else None,
                "samples": df[col].astype(str).head(3).tolist()
            }
        
        prompt = f"""
        For each column below, determine the optimal SQL Server data type:
        
        {json.dumps(column_info, indent=2)}
        
        Use the map_sql_datatype tool for each column.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a database schema expert. Use tools to map datatypes."},
                {"role": "user", "content": prompt}
            ],
            tools=self.tools,
            tool_choice="auto"
        )
        
        return self._process_tool_response(response, column_info)
    
    def _process_tool_response(self, response, context):
        """Process tool call responses"""
        result = {
            "tool_calls": [],
            "final_response": "",
            "analysis": context
        }
        
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                result["tool_calls"].append({
                    "function": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                })
        
        result["final_response"] = response.choices[0].message.content
        return result