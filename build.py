import os
import subprocess


# microrts_path = microrts_path = os.path.join(os.getcwd(), "microrts")
# print(f"removing {microrts_path}/microrts.jar...")
# if os.path.exists(f"{microrts_path}/microrts.jar"):
#     os.remove(f"{microrts_path}/microrts.jar")
# print(f"building {microrts_path}/microrts.jar...")
root_dir = os.getcwd()
print(root_dir)
subprocess.run(["bash", "build.sh", "&>", "build.log"], cwd=f"{root_dir}")