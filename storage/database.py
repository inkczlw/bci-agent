"""数据库引擎与 session 管理。

SQLAlchemy ORM 抽象层。底层默认 SQLite，切 PostgreSQL 只需改 DB_URL。
设计原则：所有数据库操作通过 get_session() 获取 session，用完自动关闭。
"""

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

# 默认 SQLite，存在项目 data/ 目录下
# 切 PostgreSQL: "postgresql://user:pass@localhost:5432/bci_agent"
_DB_DIR = Path(__file__).parent.parent / "data"
_DB_DIR.mkdir(exist_ok=True)
DB_URL = f"sqlite:///{_DB_DIR / 'bci_agent.db'}"

_engine = None
_SessionFactory = None


def get_engine():
    """获取或创建数据库引擎（单例）。"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DB_URL,
            echo=False,  # True 打印所有 SQL，debug 用
            pool_pre_ping=True,  # 连接复用前检测是否存活
            # SQLite 特有：启用 WAL 模式，允许并发读
            connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {},
        )
        # SQLite 默认不启用外键约束，手动开启
        if "sqlite" in DB_URL:

            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    return _engine


def get_session_factory():
    """获取 session 工厂（单例）。"""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


@contextmanager
def get_session():
    """获取数据库 session 的 context manager。

    用法：
        with get_session() as session:
            session.add(record)
            # 自动 commit，异常时自动 rollback
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """创建所有表（幂等操作，已存在的表不会被覆盖）。"""
    from storage.models import Base

    Base.metadata.create_all(get_engine())
    print(f"[DB] 数据库初始化完成: {DB_URL}")
