# --- User-configurable variables --------------------------------------------
REMOTE_IP      := "203.0.113.42"      # replace with the real host
REMOTE_USER    := "alice"             # your SSH user

# Change these only if you keep “results” somewhere else
LOCAL_RESULTS  := "./results"
REMOTE_RESULTS := "/workspace/results"

# ---------------------------------------------------------------------------
# All recipes run in `bash -eu` so failures stop the task
set shell := ["bash", "-eu", "-o", "pipefail"]

# Upload the local `results/` tree to the remote machine.
#  • -a : archive (preserves perms, times, symlinks, etc.)
#  • -v : verbose
#  • -z : compress while in transit
#  • --progress : live progress meter
upload-results:
    rsync -avz --progress --ignore-existing \
      {{LOCAL_RESULTS}}/ \
      {{REMOTE_USER}}@{{REMOTE_IP}}:{{REMOTE_RESULTS}}

upload-results-force:
    rsync -avz --progress \
      {{LOCAL_RESULTS}}/ \
      {{REMOTE_USER}}@{{REMOTE_IP}}:{{REMOTE_RESULTS}}

# Download the remote `results/` tree into the current directory,
# creating or updating only the files you do *not* already have.
#  • --ignore-existing ensures existing local files are left untouched
download-results:
    rsync -avz --progress --ignore-existing \
      {{REMOTE_USER}}@{{REMOTE_IP}}:{{REMOTE_RESULTS}}/ \
      .
