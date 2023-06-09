import re,hypothesis as hy, hypothesis.strategies as st, typing, inspect, collections, random
import numpy as np, pandas as pd, polars as pl
from google.cloud import bigquery as bq

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

# Some stats related funcs
def isiterable(x): return isinstance(x, (list, set, tuple, str, np.ndarray))
def isnumeric(x): return isinstance(x, (int, float, complex)) ## TODO (vijay): might need to include decimal.Decimal
def avg(xs: typing.Iterable) -> float:
  assert isiterable(xs), "Not an iterable"
  assert all(isnumeric(x) for x in xs), "Non numeric value found"
  return sum(xs)/len(xs)
mean = average = avg
def nrange(xs: typing.Iterable) -> typing.Tuple: # mnemonic: numeric range?
  assert isiterable(xs)
  assert all(isnumeric(x) for x in xs), "Non numeric value found"
  return min(xs),max(xs)
def cumsum(xs: typing.Iterable) -> typing.Iterable:
  assert isiterable(xs)
  assert all(isnumeric(x) for x in xs), "Non numeric value found"
  s=[sum(xs[:i+1]) for i in range(len(xs))]
  assert sum(xs) == s[-1]
  return type(xs)(s) ## does not work with np.ndarrays! Use cumsum(list(ndarr)) instead
def freq(xs: typing.Iterable) -> collections.Counter:
  """
  >>> freq('mississippi') # similar to pandas.value_counts
  >>> freq(repeat((1,2,3),2))
  >>> freq([(1,),(1,)])
  >>> freq([[1],[1]]) ## TypeError! Values have to be hashable!
  >>> freq(repeat([1,2,3],2)) # TypeError! Values have to be hashable!
  """
  assert isiterable(xs), "Not an iterable"
  return collections.Counter(xs)

# Some text related funcs
def squote(x): return f"'{x}'"
def dquote(x): return f'"{x}"'
singlequote,doublequote=squote,dquote
def abbrev(xs: typing.Iterable, n: int = 3) -> typing.Iterable:
  """
  >>> abbrev('vijay',3) # 'vij'
  >>> abbrev([[1,2,3,4],[1,2,3],'vijay',(1,2,3,4)],3) # [[1,2,3],[1,2,3],'vij',(1,2,3)]
  >>> abbrev([1,2,3,4,5,6,7],3) # AssertionError!
  >>> abbrev(4,3) # AssertionError!
  """
  assert isiterable(xs), "Not an iterable"
  assert type(n) == int and n > 0
  if type(xs) == str: return xs[:n]
  assert all(isiterable(x) for x in xs), "Not all items are iterable...so cannot take abbrev!"
  return type(xs)([x[:n] for x in xs])

# Useful in pandas/dask and xarray indexing
ALL = slice(None,None,None)
# Below only work for loc/iloc indexers!
def every_nth(n: int):
  assert isinstance(n, int) and n > 0
  return slice(None,None,n)

def repeat(x, n: int = 1) -> typing.List:
  """
  >>> repeat([1,2,3],2) # [[1,2,3],[1,2,3]]
  >>> repeat(4,2) # [4,4]
  >>> ''.join(repeat('abc',4)) == 'abc'*4 # True
  """
  assert type(n) == int and n > 0
  return [x for _ in range(n)]

def genrandstr(n: int = 5) -> str:
  " >>> [genrandstr(random.randint(2,8)) for _ in range(random.randint(5,10))] "

  assert type(n) == int
  if n == 0: return '' # There can be only one kind of a string with len 0!

  assert n > 0, f'invalid arg {n}'

  chars=[chr(ord('A')+i) for i in range(26)] + [chr(ord('a')+i) for i in range(26)] # + [chr(ord('0')+i) for i in range(10)]
  idx = [random.randint(0,len(chars)) for _ in range(n)]
  return ''.join(chars[i%len(chars)] for i in idx)

def print_source(obj) -> None:
  """ interesting function to find out how stuff is defined in python. check out print_source(print_source) ! """
  import inspect
  try:
      src = inspect.getsource(obj)
  except TypeError:
      src = f"src {str(obj)} of built-in module, class, or function unavailable"
  print(src)


def pandas_dataframes() -> typing.Optional[pd.DataFrame]:
  frames = [(o,globals()[o]) for o in globals() if type(globals()[o]) == pd.DataFrame and o[0] != '_']
  if len(frames) == 0:
    print("No pd.DataFrame found in globals().", file=sys.stderr)
    return None
  result = pd.DataFrame({'dataframes': [n for n,_ in frames],
                         'shape': [d.shape for _,d in frames],
                         'cols': [d.columns.array.tolist() for _,d in frames],})
  return result

def polars_dataframes() -> typing.Optional[pl.DataFrame]:
  frames = [(o,globals()[o]) for o in globals() if type(globals()[o]) == type(pl.DataFrame()) and o[0] != '_']
  if len(frames) == 0:
    print("No pl.DataFrame found in globals().", file=sys.stderr)
    return None
  result = pl.DataFrame({'dataframes': [n for n,_ in frames],
                         'shape': [d.shape for _,d in frames],
                         'cols': [d.columns for _,d in frames],})
  return result

BQParam = typing.Union[bq.ScalarQueryParameter, bq.ArrayQueryParameter, bq.StructQueryParameter]
def gcp_to_df(qry: str, params:typing.List[BQParam] = [], PROJECT:str = '') -> pd.DataFrame:
  ## See:
  ##   https://github.com/googleapis/python-bigquery/blob/main/samples/client_query_w_array_params.py
  ##   https://github.com/googleapis/python-bigquery/blob/main/samples/client_query_w_named_params.py
  ##   https://github.com/googleapis/python-bigquery/blob/main/samples/client_query_w_struct_params.py
  assert PROJECT != '', f"Cannot have empty PROJECT"
  if len(params) > 0:
    params_in_qry = [p[1:] for p in re.findall(r"(@[a-zA-Z][a-zA-Z0-9]*)", qry)]
    params_names = [p.name for p in params]
    assert set(params_names).intersection(set(params_in_qry)) == set(params_in_qry), f"Params in query missing from the params arg: {set(params_in_qry) - set(params_names)}"
  job_config = bq.QueryJobConfig(query_parameters = params)
  client = bq.Client(project=PROJECT)
  return client.query(qry, job_config=job_config).to_dataframe()


## some aliases ... especially useful in repl
get_source = get_src = print_src = print_source
q=quit
