import re, typing, inspect, collections, random, sys, dataclasses as dc,math
import numpy as np, pandas as pd
import functools,operator
try: import hypothesis as hy, hypothesis.strategies as st
except ModuleNotFoundError as e: print(f"ERROR: {e}",file=sys.stderr)

try: import polars as pl
except ModuleNotFoundError as e: print(f"ERROR: {e}",file=sys.stderr)

try: from google.cloud import bigquery as bq
except ModuleNotFoundError as e: print(f"ERROR: {e}",file=sys.stderr)

try: import duckdb as ddb
except ModuleNotFoundError as e: print(f"ERROR: {e}", file=sys.stderr)

try: import pyarrow as pa
except ModuleNotFoundError as e: print(f"ERROR: {e}", file=sys.stderr)

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
##def isiterable(x): return isinstance(x, (list, set, tuple, str, np.ndarray, range, pd.Series, pd.DataFrame))
def isiterable(x): return '__iter__' in dir(x)
def isnumeric(x): return isinstance(x, (int, float, complex)) ## TODO (vijay): might need to include decimal.Decimal
def avg(xs: collections.abc.Iterable) -> float:
  assert isiterable(xs), "Not an iterable"
  assert all(isnumeric(x) for x in xs), "Non numeric value found"
  return sum(xs)/len(xs)
mean = average = avg
def nrange(xs: collections.abc.Iterable) -> typing.Tuple: # mnemonic: numeric range?
  assert isiterable(xs)
  assert all(isnumeric(x) for x in xs), "Non numeric value found"
  return min(xs),max(xs)
def cumsum(xs: collections.abc.Iterable) -> collections.abc.Iterable:
  assert isiterable(xs)
  assert all(isnumeric(x) for x in xs), "Non numeric value found"
  s=[sum(xs[:i+1]) for i in range(len(xs))]
  assert sum(xs) == s[-1]
  return type(xs)(s) ## does not work with np.ndarrays! Use cumsum(list(ndarr)) instead
def freq(xs: collections.abc.Iterable) -> collections.Counter:
  """
  >>> freq('mississippi') # similar to pandas.value_counts
  >>> freq(repeat((1,2,3),2))
  >>> freq([(1,),(1,)])
  >>> freq([[1],[1]]) ## TypeError! Values have to be hashable!
  >>> freq(repeat([1,2,3],2)) # TypeError! Values have to be hashable!
  """
  assert isiterable(xs), "Not an iterable"
  return collections.Counter(xs)
def softmax(xs,base=math.exp(1)):
  """
  Even though the most commonly used base is e, any other base (greater than 0) can be used.
  If the base is > 1 then larger values will get higher probabilities.
  If 0 < base < 1 then smaller values will get higher probabilities.

  >>> x = np.random.standard_normal(15)
  >>> softmax(x.tolist())
  >>> softmax(x.tolist(),base=0.8)
  >>> pd.DataFrame({'x':x,'softmax':sofmtax(x.tolist()),'softmax1':sofmtax(x.tolist(),base=0.8)})
  """
  assert isiterable(xs)
  assert all(isnumeric(x) for x in xs)
  exps = type(xs)(base**x for x in xs)       ## does not work with np.ndarray! Use softmax(np.random.standard_normal(15).tolist())
  return type(xs)(e/sum(exps) for e in exps)
def prop(xs: collections.abc.Iterable) -> collections.abc.Iterable:
  """
  Emulate R's prop.table or proportions function.

  >>> prop([1,1,1,1]) # [0.25,0.25,0.25,0.25]
  >>> prop([1,2,1,1]) # [0.2,0.4,0.2,0.2]
  >>> prop((1,2,3,4)) # (0.1,0.2,0.3,0.4)
  >>> prop([]) # []
  >>> prop(np.arange(1,5)) ## TypeError!
  >>> prop(np.arange(1,5).tolist()) # :-(
  >>> prop(freq(df1.select(pl.col("col1")).to_series()))
  >>> prop({"a":1,"b":1,"c":1,"d":1})
  >>> prop({"a":1,"b":1,"c":1,"d":1,"e":"tst"}) ## will raise assertion error!
  """
  assert isiterable(xs), "Not an iterable"
  if isinstance(xs,dict):
    assert all(isinstance(v,(int,float,complex)) for _,v in xs.items()),"Non numeric values"
    sv=sum(xs.values())
    ret=({k:v/sv for k,v in xs.items()})
  else:
    ret = type(xs)(x/sum(xs) for x in xs)
  return ret

# Some text related funcs
def squote(x): return f"'{x}'"
def dquote(x): return f'"{x}"'
singlequote,doublequote=squote,dquote
def abbrev(xs: collections.abc.Iterable, n: int = 3) -> collections.abc.Iterable:
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

def genrandstr(n: int = 5, lowercase=False) -> str:
  " >>> [genrandstr(random.randint(2,8)) for _ in range(random.randint(5,10))] "

  assert type(n) == int
  if n == 0: return '' # There can be only one kind of a string with len 0!

  assert n > 0, f'invalid arg {n=}'

  ## chars=[chr(ord('A')+i) for i in range(26)] + [chr(ord('a')+i) for i in range(26)] # + [chr(ord('0')+i) for i in range(10)]
  ## idx = [random.randint(0,len(chars)) for _ in range(n)]
  ## return ''.join(chars[i%len(chars)] for i in idx)
  import string,random
  chars = string.ascii_lowercase + ('' if lowercase else string.ascii_uppercase) + string.digits
  return ''.join(random.choice(chars) for _ in range(abs(n)))

def get_source(obj) -> str:
  """ interesting function to find out how stuff is defined in python. check out print_source(get_source) ! """
  def _get_src(o) -> str:
    import inspect
    try:
      src = inspect.getsource(o)
    except TypeError as e:
      src = f"src {str(o)} of built-in module, class, or function unavailable"
      print(f"{e}",file=sys.stderr)
    return src
  return _get_src(obj)

def rangealong(l: collections.abc.Iterable) -> collections.abc.Iterable:
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
    result = pl.DataFrame([(n, *d.shape, d.columns,round(d.estimated_size(unit="mb"),2)) for n,d in frames],schema=["dataframe","nrow","ncol","columns","size (mb)"], orient="row")
    return result
except NameError as e:
  print(f"ERROR: {e}",file=sys.stderr)

try:
  BQParam = typing.Union[bq.ScalarQueryParameter, bq.ArrayQueryParameter, bq.StructQueryParameter, bq.RangeQueryParameter]
  def gcp_to_df(qry: str, params:typing.List[BQParam] = [], PROJECT:str = '') -> pd.DataFrame:
    """Example usage:
    df = gcp_to_df(qry="select * from `bigquery-public-data.idc_v17.dicom_all` where StudyDate=@dt",params=[bq.ScalarQueryParameter("dt","DATE",datetime.date(2010,1,1))],PROJECT="<your-project>")
    """
    ## See:
    ##   https://github.com/googleapis/python-bigquery/blob/main/samples/client_query_w_array_params.py
    ##   https://github.com/googleapis/python-bigquery/blob/main/samples/client_query_w_named_params.py
    ##   https://github.com/googleapis/python-bigquery/blob/main/samples/client_query_w_struct_params.py
    assert PROJECT != '', f"Cannot have empty PROJECT"
    if len(params) > 0:
      params_in_qry = [p[1:] for p in re.findall(r"(@[a-zA-Z][a-zA-Z0-9_]*)", qry)]
      params_names = [p.name for p in params]
      assert (set(params_names) & set(params_in_qry)) == set(params_in_qry), f"Params in query missing from the params arg: {set(params_in_qry) - set(params_names)}"
    job_config = bq.QueryJobConfig(query_parameters = params)
    client = bq.Client(project=PROJECT)
    return client.query(qry, job_config=job_config).to_dataframe()
  def gcp_to_polars(qry: str, params:typing.List[BQParam]=[], PROJECT:str='') -> pl.DataFrame:
    ## NOTE (vijay): This does not work with Interval/Duration types! I get the error "The datatype tin (for IntervalUnit::MonthDayNanon) is still not supported in Rust implementation....see https://arrow.apache.org/rust/src/arrow_schema/ffi.rs.html
    assert PROJECT != '', f"Cannot have empty PROJECT"
    if len(params) > 0:
      params_in_qry = [p[1:] for p in re.findall(r"(@[a-zA-Z][a-zA-Z0-9_]*)", qry)]
      params_names = [p.name for p in params]
      assert (set(params_names) & set(params_in_qry)) == set(params_in_qry), f"Params in query missing from the params arg: {set(params_in_qry) - set(params_names)}"
    job_config = bq.QueryJobConfig(query_parameters = params)
    client = bq.Client(project=PROJECT)
    ## df = gcp_to_df(qry,params,PROJECT)
    ## dfp = pl.from_arrow(pa.Table.from_pandas(df)) ## NOTE (vijay): need this because pl.from_pandas(df) cannot read db_dtypes.dbdate datatype!
    ## return dfp
    df = pl.from_arrow(client.query(qry, job_config=job_config).to_arrow())
    return df
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

def df_coltypes(df: T) -> T:
  assert isinstance(df, (pd.DataFrame, pl.DataFrame))
  typ = type(df)
  if typ == pd.DataFrame:
    cols_with_attrs = [(i,c,f"{str(df[c].dtype)}",df[c].nunique(),df[c].isna().sum(),100*df[c].isna().sum()/df.shape[0]) for i,c in enumerate(df.columns)]
    ret = pd.DataFrame(cols_with_attrs, columns=["colidx", "colname", "coltype", "nunique", "numna", "pctna"])
  elif typ == pl.DataFrame:
    cols_with_attrs = [(i,c,f"{str(df[c].dtype)}",df[c].n_unique(),df[c].null_count(),100*df[c].null_count()/df.shape[0]) for i,c in enumerate(df.columns)]
    ret = pl.DataFrame( cols_with_attrs,schema=["colidx","colname","coltype","nunique","numna","pctna"])
  return ret

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


def grep[L: collections.abc.Iterable[T]](regex: str, lst: L, invert=False, ignorecase=True) -> L:
  """
  Like R's grep function...
  >>> grep("_spend", ['abc', 'xyz_spend', 'abc_spend_xyz'])
  >>> grep("_spend$", ['abc', 'xyz_spend', 'abc_spend_xyz'])
  >>> grep("_spned", ['abc', 'xyz_spend', 'abc_spend_xyz']) ## typo in regex
  >>> grep("_spend$", df.columns.tolist()) ## extract spend cols
  >>> grep("_spend$", df.columns.tolist(), invert=True) ## everything except spend cols
  >>> grep("_spend$",('abc','xyz_spend','abc_spend_xyz'))
  >>> grep("_spend",{'abc','xyz_spend','abc_spend_xyz'))

  For dict/colletions.Counter, the function filters based on key and tries to return the appropriate type.
  TODO: Ensure that this works correctly for dict like types.
  """
  assert isinstance(regex, str)
  ## assert isinstance(lst, list)
  assert isiterable(lst)
  assert all(isinstance(o, str) for o in lst)
  assert all(isinstance(o, bool) for o in (invert, ignorecase))
  flags = re.UNICODE | re.VERBOSE
  if ignorecase: flags |= re.IGNORECASE
  regexc = re.compile(regex, flags)
  if invert: return type(lst)(c for c in lst if re.search(regexc, c) is None)
  if isinstance(lst,(dict,collections.Counter)): return type(lst)({k:v for k,v in lst.items() if re.search(regexc,k) is None})
  return type(lst)(c for c in lst if re.search(regexc, c) is not None)

def gsub[L: list[str] | set[str] | tuple[str]](regex: str, repl: str, lst: str | L) -> str | L:
  """
  Like R's gsub function...
  >>> gsub("_spend$", "", df.columns.tolist())
  >>> gsub("_spend$", "", grep("_spend$", df.columns.tolist()))
  """
  assert isinstance(regex, str)
  assert isinstance(repl, str)
  ## assert isinstance(lst, (str, list))
  assert isiterable(lst)
  if isinstance(lst, (list,set,tuple)): assert all(isinstance(i, str) for i in lst)

  @functools.cache
  def _gsub(_regex: str, _repl: str, _string: str) -> str:
    regexc = re.compile(_regex, re.IGNORECASE | re.UNICODE | re.VERBOSE)
    return re.sub(regexc, _repl, _string)
  if type(lst) == type(''): return _gsub(regex, repl, lst)
  return type(lst)(_gsub(regex, repl, c) for c in lst)

P = typing.ParamSpec('P')
def negate(pred: collections.abc.Callable[P, bool]) -> collections.abc.Callable[P, bool]:
  """
  This is useful for filter. And, it is also like itertools.filterfalse

  >>> def isodd(x): return x % 2 != 0
  >>> [i for i in range(15) if isodd(i)]
  >>> [i for i in range(15) if negate(isodd)(i)]
  >>> iseven = negate(isodd)
  >>> assert [i for i in range(15) if iseven(i)] == [i for i in range(15) if not isodd(i)]
  >>> assert list(itertools.filterfalse(lambda x: x%2, range(10))) == list(filter(negate(lambda x: x%2), range(10)))
  """
  def _inner(*args: P.args, **kwargs: P.kwargs) -> bool:
    return not pred(*args,**kwargs)
  return _inner

def make_param_grid(param: dict) -> pd.DataFrame:
  """
  Trying to emulate R's expand.grid.
  R> expand.grid(x=1:5,y=12:13)
  >>> make_param_grid({'x':[1,2,3,4,5],'y':[12,13]})
  >>> make_param_grid({'chgpt_prior_scale':np.linspace(0.001,0.5,num=5).tolist(),
                       'holidays_prior_scale': np.linspace(0.01,10,num=5).tolist()})
  """
  assert isinstance(param, dict)
  assert all(isinstance(v, list) for v in param.values())
  import itertools
  df = pd.DataFrame(itertools.product(*param.values()),columns=param)
  assert df.shape == (np.prod([len(_) for _ in param.values()]), len(param))
  return df

def args(o: object) -> inspect.Signature:
  """
  Trying to emulate R's args function.

  >>> args(args)
  >>> args(prophet.Prophet)
  >>> args(str) ## does not work
  >>> [(o,args(getattr(builtins,o))) for o in dir(builtins) if callable(getattr(builtins,o)) and args(getattr(builtins,o)) is not None]

  Does not work for builtin functions like str and int (possibly others too).
  """
  assert callable(o), "Need a callable object"
  try:
    sig = inspect.signature(o)
  except ValueError as e:
    print(f"{e}", file=sys.stderr)
    sig = None
  return sig

def splitarray(xs: collections.abc.Iterable[T], stride:int) -> collections.abc.Iterable[T]:
  """
  Generates ragged array by splitting iterable into chunks where majority of the elements are of length stride.

  Emulates K's excellent idiom: `0N 3#!10`
  >>> l10 = list(range(10))
  >>> l9 = l10[:9]
  >>> s = "abcdefghij"
  >>> splitarray(l10,3) # [[0,1,2],[3,4,5],[6,7,8],[9]]
  >>> splitarray(tuple(l9),3) # ((0,1,2),(3,4,5),(6,7,8))
  >>> splitarray(s,3) # ['abc','def','ghi','j']
  >>> splitarray(s[:9],3) # ['abc','def','ghi']
  >>> splitarray([],15) # ought to handle strange cases correctly...
  >>> splitarray("abc",5) # ["abc"]
  >>> splitarray("abc",1) # ["a","b","c"]
  >>> splitarray(tuple(l9),2) ((0,1),(2,3),(4,5),(6,7),(8,))
  """
  if xs=='': return ['']
  assert stride > 0, "Cannot have -ve stride!"
  idxs = [(stride*_,(stride*_)+stride) for _ in range(len(xs)//stride)] + ([(stride*(len(xs)//stride),None)] if len(xs)%stride!=0 else [])
  if type(xs) == str:
    ret =          [type(xs)(xs[_[0]:_[-1]]) for _ in idxs]
  else:
    ret = type(xs)([type(xs)(xs[_[0]:_[-1]]) for _ in idxs])
  return ret

## @hy.given(st.lists(st.integers()|st.floats())|st.text(),st.integers(min_value=1))
## def test_splitarray(xs,n):
##   ## print(f"(xs,n) => {xs,n}") ## To see that hypothesis is actually running this test!
##   assert xs == functools.reduce(lambda a,b: a+b, splitarray(xs,n),type(xs)())

def idir():
  """
  dir alternative for IPython.
  Since IPython caches input/output history in hidden variables, dir() creates a lot of noise. This function reduces this noise.
  Type ? in the IPython prompt to learn about IPython's history/caching mechanism.
  """
  spls = ('_','__','___','_i','_ii','_iii','_ih','_oh','_dh')
  return [_ for _ in globals() if not _ in spls and not re.match('^_[io]?[0-9]+$',_)]

def monthnames():
  """
  Trying to emulate R's month.name and month.abb constants
  >>> monthnames()
  >>> abbrev(monthnames())
  >>> pl.DataFrame({"month":monthnames(),"abbr":abbrev(monthnames())})
  """
  return 'January','February','March','April','May','June','July','August','September','October','November','December'
def daynames(): return 'Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'
def monthdays(isleapyear=False): return 31,28+isleapyear,31,30,31,30,31,31,30,31,30,31

def togglesqlcomment(s):
  assert isinstance(s,(type(""),)), "Need a str"
  if not s: return s ## s=="" or s==None
  return s[3:] if s[:3] == '-- ' else f"-- {s}"


## some aliases ... especially useful in repl
def print_source(o): print(get_source(o))
print_src, get_src = print_source, get_source
q=quit
