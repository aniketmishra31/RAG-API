from supabase import create_client,Client
import os

SUPABASE_URL=os.getenv("SUPABASE_URL")
SUPABASE_KEY=os.getenv("SUPABASE_KEY")
db: Client=create_client(SUPABASE_URL,SUPABASE_KEY)
print("DB is connected")