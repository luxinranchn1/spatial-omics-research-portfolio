"""Read-only GPU inventory. This script never launches or terminates jobs."""
import csv,io,json,shutil,subprocess
if not shutil.which('nvidia-smi'):
    print(json.dumps({'GPU_available':False,'reason':'nvidia-smi not found'}))
else:
    run=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    print(json.dumps([dict(zip(['index','name','total_MiB','free_MiB','utilization_percent'],[s.strip() for s in row])) for row in csv.reader(io.StringIO(run.stdout))],ensure_ascii=False,indent=2))
