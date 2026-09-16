from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    app_name: str = "Pickleball Expert Assistant"
    version: str = "0.1.0"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    database_url: str = (
        "postgresql+asyncpg://pickleball:pickleball@localhost:5432/pickleball"
    )
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"      
    llm_fast_model: str = "deepseek-chat"  
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    harness_max_steps: int = 8             
    harness_timeout_seconds: float = 150.0  
    harness_verify_required: bool = True   
    harness_max_tool_calls: int = 4        

    tavily_api_key: str = ""
    tavily_search_depth: str = "basic"   # basic=快速便宜 | advanced=更深更慢
    tavily_max_results: int = 5          # 单次搜索返回条数（1-10）

    vision_model: str = ""
    vision_base_url: str = ""
    vision_api_key: str = ""

    # ---- MCP 外部工具服务器（JSON 数组；解析失败只告警，不阻断启动）----
    mcp_servers: str = "[]"

    # ---- 记忆与检索 ----
    knowledge_dir: str = "knowledge"       
    skills_dir: str = "skills"            
    memory_window_size: int = 12           
    memory_summary_trigger: int = 20       
    retrieval_top_k: int = 4               
    retrieval_plan_enabled: bool = True    
    retrieval_evidence_limit: int = 6      
    retrieval_rerank_enabled: bool = True  
    retrieval_rerank_candidates: int = 8   
    embedding_dim: int = 4096              

    memory_reflective_enabled: bool = True 
    memory_reflective_top_k: int = 3      
    memory_reflective_recent: int = 100    

    chat_execution_mode: str = "auto"     
    stream_heartbeat_seconds: float = 15.0  
    chat_max_total_seconds: float = 300.0   
    worker_heartbeat_ttl: int = 10        

    cost_input_per_mtok: float = 0.27
    cost_output_per_mtok: float = 1.10

    @property
    def knowledge_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / self.knowledge_dir

    @property
    def skills_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / self.skills_dir

@lru_cache
def get_settings() -> Settings:
    return Settings()
