import re,hypothesis as hy, hypothesis.strategies as st

def fix_colnames(colname: str, normalize_adjacent_uppers: bool = True) -> str:
  """
    Similar to the algorithm described at https://pandoc.org/MANUAL.html#extension-auto_identifiers
    SEP is '_' ## '-' is inadvisable because I cannot use column name in Python's dot notation!

    1. Replace all uppercase characters to SEP + lowercase equivalent
    2. Replace all non-alphanumeric characters to SEP
    3. Replace multiple instances of SEP to a single instance of SEP
    4. Remove *all* beginning and trailing instances of SEP ## Not needed but python considers anything starting with _ as hidden....


    It is common to have ID, DOB, YYYY, MM as part of colnames. Using the rules above makes weird colnames. So, instead use a flag to see how to handle this.
    If the `normalize_adjacent_uppers` is True then just keep the first character as upper and everything else as lower.
    Using `re` to do this. TODO: needs a less fragile solution...
  """
  SEP = '_'

  assert len(colname) != 0, "Colname cannot be empty"

  ## TODO: hacky! needs more thought
  if normalize_adjacent_uppers:
      pat = re.compile('[A-Z][A-Z]+')
      colname = pat.sub(lambda x: x.group(0).title(), colname)

  fixed_colname = re.sub(r'([A-Z])', rf"{SEP}\1", colname).lower() # step 1
  fixed_colname = re.sub(r'[^A-Za-z0-9]', SEP, fixed_colname)   # step 2
  fixed_colname = re.sub(f"{SEP}+", f"{SEP}", fixed_colname)    # step 3
  fixed_colname = re.sub(f"^{SEP}+|{SEP}+$", "", fixed_colname) # step 4

  ## Hypothesis failed at fix_colnames(':') and fix_colnames('0')! Hypothesis is awesome!!
  if fixed_colname == '': fixed_colname = 'tmp_col'
  if fixed_colname[0] in '0123456789': fixed_colname = f'c_{fixed_colname}'
  return fixed_colname


## @hy.settings(max_examples=500) # more thorough but slower
@hy.given(st.text(min_size=1))
def test_colname_fixer(s: str):
  ## TODO: update this with asserts capturing failures that I'm bound to run into
  s1 = fix_colnames(s)
  msg = f"{s} ==> {s1}"

  assert ' ' not in  s1, f"Rule 2 failed! {msg}"
  assert len(s1) > 0, f"Cannot have empty colname! {msg}"
  assert all(ch not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for ch in s1), f"Rule 1 failed! {msg}"
  assert '_' != s1[0] or '_' != s1[-1], f"Rule 4 failed! {msg}"
  assert s1[0] not in '0123456789', f"Cannot begin with a number! {msg}"
  assert '__' not in s1, f"Rule 3 failed! {msg}"
  assert '.' not in s1, f"Rule 2 failed! {msg}"
  assert re.match('_[^_]_[^_]', s1) is None, f"Some special cols...{s} ==> {s1}"
