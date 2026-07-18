##
## $ micromamba run --name lgb python update-mamba-env.py
##
import argparse, subprocess, sys, json, typing

def runcmd(cmd: list[str]) -> str:
    assert isinstance(cmd, list) and all(isinstance(_, str) for _ in cmd)

    res = subprocess.run(cmd, capture_output=True, check=True)
    assert res.returncode==0, f"{res.stderr}"

    stdout = res.stdout.decode()
    return stdout

def get_dependencies(env: str) -> dict[str, list[str]]:
    cmd = ["micromamba", "env", "export", "--name", env, "--from-history", "--json"]
    j = json.loads(runcmd(cmd))
    assert "dependencies" in j

    deps = j["dependencies"]
    env_deps = { "regular_pkgs": [_ for _ in deps if isinstance(_, str)] }

    pip = list(filter(lambda _: isinstance(_, dict) and 'pip' in _, deps))[0:]
    if pip:
        env_deps["pip_pkgs"] = [_.split("=")[0] for _ in pip[0]["pip"]]
    return env_deps

def update_mambaenv(env):
    deps = get_dependencies(env)
    assert "regular_pkgs" in deps

    drop_env_cmd = ["micromamba", "env", "remove", "--name", env, "--yes"]
    print("running...", drop_env_cmd)
    runcmd(drop_env_cmd)

    update_pkgs_cmd = ["micromamba", "create", "--name", env, "--yes", *deps["regular_pkgs"]]
    print("running...", update_pkgs_cmd)
    runcmd(update_pkgs_cmd)

    if "pip_pkgs" in deps:
        update_pips_cmd = ["micromamba", "run", "--name", env, "pip", "install", *deps["pip_pkgs"]]
        print("running...", update_pips_cmd)
        runcmd(update_pips_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="update-mamba-env")
    parser.add_argument("envname")
    args = parser.parse_args()
    update_mambaenv(args.envname)

