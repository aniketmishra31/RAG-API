from supabase import create_client,Client
import os

SUPABASE_URL=os.getenv("SUPABASE_URL")
SUPABASE_KEY=os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
db: Client=create_client(SUPABASE_URL,SUPABASE_KEY)
print("DB is connected")