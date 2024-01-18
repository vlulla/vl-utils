import re, typing, inspect, collections, random, sys, dataclasses as dc
import numpy as np, pandas as pd
import functools,operator
try:
  import hypothesis as hy, hypothesis.strategies as st
except ModuleNotFoundError as e:
  print(f"ERROR: {e}",file=sys.stderr)

try:
  import polars as pl
except ModuleNotFoundError as e:
  print(f"ERROR: {e}",file=sys.stderr)

try:
  from google.cloud import bigquery as bq
except ModuleNotFoundError as e:
  print(f"ERROR: {e}",file=sys.stderr)

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


T = typing.TypeVar("T")
def identity(x: T) -> T: return x ## surprisingly useful!

## ## @hy.settings(max_examples=500) # more thorough but slower
## @hy.given(st.text(min_size=1))
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
def isiterable(x): return isinstance(x, (list, set, tuple, str, np.ndarray, range, pd.Series, pd.DataFrame))
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
  except TypeError as e:
      src = f"src {str(obj)} of built-in module, class, or function unavailable"
      print(f"{e}",file=sys.stderr)
  print(src)

def rangealong(l: typing.Iterable) -> typing.Iterable:
  """
  Like R's seq_along! But works differently for pd.DataFrame!
  >>> rangealong(pd.DataFrame({'a':range(10),'b':[i*10 for i in range(10)]})) ## range(0,10)
  R> seq_along(data.frame(a=1:10,b=10*(1:10))) ## [1] 1 2

  In R: dataframe *is a list* of columns! ## R> is.list(data.frame(a=1:5)) ## TRUE
  In python: dataframe is a collection of rows (or a tuple) of columns!
  """
  assert isiterable(l), f"Argument ({l}) is not iterable!"
  return range(len(l))


def pandas_dataframes() -> typing.Optional[pd.DataFrame]:
  frames = [(o,globals()[o]) for o in globals() if type(globals()[o]) == pd.DataFrame and o[0] != '_']
  if len(frames) == 0:
    print("No pd.DataFrame found in globals().", file=sys.stderr)
    return None
  ## result = pd.DataFrame([(n, d.memory_usage(deep=True).sum(), *d.shape, d.columns.array.tolist()) for n,d in frames],
  ##                       columns=["dataframe","memsize","nrow","ncol","columns"])
  result = pd.DataFrame([(n, *d.shape, d.columns.array.tolist()) for n,d in frames],
                        columns=["dataframe","nrow","ncol","columns"])
  return result

try:
  def polars_dataframes() -> typing.Optional[pl.DataFrame]:
    frames = [(o,globals()[o]) for o in globals() if type(globals()[o]) == type(pl.DataFrame()) and o[0] != '_']
    if len(frames) == 0:
      print("No pl.DataFrame found in globals().", file=sys.stderr)
      return None
    result = pl.DataFrame([(n, *d.shape, d.columns) for n,d in frames],schema=["dataframe","nrow","ncol","columns"])
    return result
except NameError as e:
  print(f"ERROR: {e}",file=sys.stderr)

try:
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
except NameError as e:
  print(f"ERROR: {e}",file=sys.stderr)

def calculate_woe(df: pd.DataFrame, feature: str, target: str, zeroadjust=True) -> typing.Tuple[pd.DataFrame, float]:
  ## https://documentation.sas.com/doc/en/vdmmlcdc/8.1/casstat/viyastat_binning_details02.htm
  ## https://www.google.com/search?q=weight+of+evidence

  assert feature in df.columns, f"{feature} not in {df.columns.tolist()}"
  assert target in df.columns, f"{feature} not in {df.columns.tolist()}"

  uniq_feats = df[feature].unique()
  dset = pd.DataFrame([{'FeatVal': f"{feature}-{featval}", 'N': (df[feature]==featval).sum(),
                        'NonEvent': ((df[feature]==featval) & (df[target]==0)).sum(),
                        'Event': ((df[feature]==featval) & (df[target]==1)).sum()}
                       for featval in uniq_feats])

  TotNonEvent = dset['NonEvent'].sum()
  TotEvent = dset['Event'].sum()
  assert TotNonEvent == (df[target]==0).sum(), "NonEvent numbers don't match!"
  assert TotEvent == (df[target]==1).sum(), "Event numbers don't match!"

  x = 0.5 if zeroadjust == True else 0

  dset['WoE'] = np.log(((dset['NonEvent'] + x)/TotNonEvent)/((dset['Event'] + x)/TotEvent))
  iv = (((dset['NonEvent']/TotNonEvent) - (dset['Event']/TotEvent)) * dset['WoE']).sum()
  return dset.loc[:,['FeatVal','WoE']], iv

def df_coltypes(df: pd.DataFrame) -> pd.DataFrame:
  cols_with_attrs = [(i,c,f"{str(df[c].dtype)}",df[c].nunique(),df[c].isna().sum(),100*df[c].isna().sum()/df.shape[0]) for i,c in enumerate(df.columns)]
  return pd.DataFrame(cols_with_attrs, columns=["colidx", "colname", "coltype", "nunique", "numna", "pctna"])

def make_dataclass_from_df(df: pd.DataFrame, name_of_dataclass: str="DF"):
  """
    >>> d = pd.DataFrame({f"col{i:02}": np.random.randn(10)*np.random.randint(50) for i in range(np.random.randint(5,25))})
    >>> D = make_dataclass_from_df(d, "D")
    >>> _d = D(*d.iloc[0])
    >>> _d
    >>> places = pd.DataFrame({"lat": [28.499163,34.044292,-33.889114], "lon": [34.518745,-118.904991,151.225204], "name": ["Dahab Freedivers, Egypt", "Barbie Dream House, Malibu, CA, USA", "Sydney Football Stadium, NSW, Australia"]})
    >>> Place = make_dataclass_from_df(places, "Place")
    >>> barbiehouse = Place(*places.iloc[[1],:].iloc[0]) ## for dataframe
    >>> barbiehouse = Place(*places.iloc[1  ,:]        ) ## for series
    >>> ## barbiehouse = Place(*places.iloc[[1],:]     ) ## NOTE: wrong!
    >>> barbiehouseDf = pd.DataFrame({k:[v] for k,v in dc.asdict(barbiehouse).items()})
  """
  assert df.shape[1]>0, f"df.shape appears strange. {df.shape}"

  import dataclasses as dc
  return dc.make_dataclass(name_of_dataclass, [(str(c).replace(' ','_'), df[c].dtypes.type) for c in df.columns])

def get_callables_for(o: typing.Any) -> typing.Dict[str,typing.Callable]:
  """
    >>> pd_funcs = (get_callables_for(pd) | get_callables_for(pd.DataFrame) | get_callables_for(pd.Series))
    >>> df = pd.DataFrame([(name,inspect.signature(func),len(inspect.signature(func).parameters))
                           for name,func in pd_funcs.items() if type(func)!=type],columns=["funcname","sig","nparams"])
    >>> all_funcs = functools.reduce(operator.or_,[get_callables_for(globals()[m]) for m in dir() if inspect.ismodule(globals()[m])], {})
    >>> df = pd.DataFrame([(name,inspect.signature(func),len(inspect.signature(func).parameters))
                           for name,func in all_funcs.items() if type(func)!=type],columns=["funcname","sig","nparams"])
    >>> df.sort_values("nparams",ascending=False).iloc[:5,:]
    >>> lgb_callables = get_callables_for(lgb)
    >>> ldf = pd.DataFrame([(fn,inspect.signature(fc),len(inspect.signature(fc).parameters),str(type(fc)))
                            for fn,fc in lgb_callables.items() if type(fc)!=type],columns=["callablename","sig","nparams","callabletype"])

    Many of the above have stopped working! Python has lots of types that cannot be inspected...and i do not yet know how to filter these types out!
    >>> collections.Counter(type(v) for _,v in all_funcs.items())
    >>> cannot_be_inspected = set(); type_not_supported = set()
    >>> for n,v in all_funcs.items():
    ...   try:
    ...     inspect.signature(v)
    ...   except ValueError:
    ...     cannot_be_inspected.add(v)
    ...   except TypeError:
    ...     type_not_supported.add(v)
    >>> ## inspect.signature(random.choice(tuple(cannot_be_inspected)))
  """
  return {f"{o.__name__}.{fname}": getattr(o,fname) for fname in dir(o) if callable(getattr(o,fname))}

def grid(axis="both"):
  ## Neat idea from https://github.com/norvig/pytudes/blob/main/ipynb/BikeCode.ipynb
  import matplotlib.pyplot as plt
  ## plt.rcParams['figure.figsize'] = (12, 6)
  plt.minorticks_on()
  plt.grid(which="major", ls="-", alpha=3/4, axis=axis)
  plt.grid(which="minor", ls=":", alpha=1/2, axis=axis)

## some aliases ... especially useful in repl
get_source = get_src = print_src = print_source
q=quit
