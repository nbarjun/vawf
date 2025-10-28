pip install -r requirements.txt
python vawf_workflow.py
git config user.name "github-actions"
git config user.email "github-actions@github.com"
git add outputs
git commit -m "Updated VAWF"
git push