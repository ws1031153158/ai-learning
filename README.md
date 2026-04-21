## 框架        FastAPI
APIRouter      模块化路由（auth/watchlist/analysis/chat）  
Depends        依赖注入（鉴权/数据库）   
Pydantic       请求/响应模型验证  
Header         统一读取 Authorization  
JSONResponse   统一错误响应  
async/await    异步接口
## 持久化      SQLite + SQLAlchemy ORM + Mem0
User           用户表（id/username/email/phone/password_hash）  
WatchlistItem  持仓表（user_id/code/name/type）  
Position       买卖记录表（watchlist_id/action/price/shares/date）  
UserPreference 用户偏好表（风险偏好/投资风格/总资产）  
Memory         AI记忆表（Mem0管理）  
## 鉴权        JWT（python-jose）
注册/登录  →  生成 JWT Token  
前端存储   →  DataStore  
每次请求   →  Header: Bearer <token>  
后端验证   →  AuthService.get_current_user()  
## AI          DeepSeek API（OpenAI 兼容）
定义工具    FC_TOOLS（JSON Schema）  
工具映射    TOOL_MAP（函数名 → 函数）  
循环调用    _run_fc_loop（最多5轮）  
自动决策    AI 决定何时调用工具、何时返回  
## Agent       CrewAI（多 Agent 协作）
数据收集师   获取价格/新闻/资金流向  
技术分析师   分析技术面  
基本面分析师  分析财务指标  
风险评估师   评估投资风险  
首席分析师   整合输出完整报告  

Process.sequential  顺序执行  
context             任务间数据传递  
## 部署        Nginx + Gunicorn + Systemd
Nginx       代理服务器  
Gunicorn    配合 Nginx 处理高并发的 HTTP 请求分发
Systemd     项目生命周期绑定服务器  
