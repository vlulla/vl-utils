-- Some functions I find useful in sql explorations.
--
-- See
-- UserGuide: https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions
-- Reference: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_function_statement

create or replace function round2(x any type) as ( round(x, 2) );
create or replace function round3(x any type) as ( round(x, 3) );
create temp function median(arr any type) as ( (select percentile_cont(x, 0.5) over() from unnest(arr) as x limit 1) );
ASSERT median(ARRAY<float64>[1,2,3,4,5,6,7,8,9,10]) = 5.5;
-- R's quantile
create temp function quantile(arr any type) as ( [(select percentile_cont(x, 0) over() from unnest(arr) as x limit 1),
                                                  (select percentile_cont(x, 0.25) over() from unnest(arr) as x limit 1),
                                                  (select percentile_cont(x, 0.5) over() from unnest(arr) as x limit 1),
                                                  (select percentile_cont(x, 0.75) over() from unnest(arr) as x limit 1),
                                                  (select percentile_cont(x, 1) over() from unnest(arr) as x limit 1)]);
-- ASSERT are_arrays_equal(quantile([1.0,2,3,4,5,6,7,8,9,10]), [1,3.25,5.5,7.75,10]); -- see below for definition of are_arrays_equal udf
create temp function IQR(arr any type) as ( quantile(arr)[safe_ordinal(4)] - quantile(arr)[safe_ordinal(2)]);
ASSERT IQR([7,7,31,31,47,75,87,115,116,119,119,155,177]) = 88;


-- https://webspace.science.uu.nl/~gent0113/calendar/isocalendar.htm for iso calendar algorithms!

-- [dw]oy: [day/week]-of-year
create or replace function iso_yyyy_woy(x any type) as (cast(format_date("%G%V", cast(x as date)) as int64) ); -- https://en.wikipedia.org/wiki/ISO_week_date
assert iso_yyyy_woy('2023-05-01') = 202318; assert mod(iso_yyyy_woy(current_date()),100) between 1 and 53;
create or replace function iso_yyyy_doy (x any type) as (cast(format_date("%G%J", cast(x as date)) as int64) );
assert iso_yyyy_doy ('2023-01-02') = 2023001;
assert iso_yyyy_doy('2023-01-01') = 2022364; -- !!
-- select d,iso_yyyy_woy(d) isowk,iso_yyyy_doy(d) isoday from unnest(generate_date_array('2023-01-01','2023-12-31',interval 1 day)) as d;
create or replace function date_diff_to_seconds(dt1 date, dt2 date) as (
  extract(day from dt1 - dt2) * 24 * 60 * 60
);
assert date_diff_to_seconds('2010-07-07','2010-07-07') = 0;
assert date_diff_to_seconds('2008-12-25','2010-07-07') = -559*24*60*60;
assert date_diff_to_seconds('2010-07-07','2008-12-25') =  559*24*60*60;

-- See some thorough discussion at https://www.gnu.org/software/coreutils/faq/coreutils-faq.html#The-date-command-is-not-working-right_002e
create or replace function yyyy_woy(x any type) as (cast(format_date('%Y%W', cast(x as date)) as int64) );
assert mod(yyyy_woy(current_date()),100) between 0 and 53;
create or replace function yyyy_doy(x any type) as (cast(format_date('%Y%j', cast(x as date)) as int64) );
assert mod(yyyy_doy(current_date()),1000) between 1 and 366;

create or replace function datediff_businessdays(d1 date, d2 date) returns int as (
  -- This ought to reconcile with pd.bdate_range!
  (with _ as (select dt from unnest(generate_date_array(d1,d2,interval 1 day)) as dt)select sum(cast(extract(dayofweek from dt) in (2,3,4,5,6) as int)) from _)
);


-- NOTE (vijay): BQ does not allow creating this without project/dataset_id info. So, if you wish to create this in a temporary session follow this:
-- 1. Open a new tab in BQ console and enable "Session Mode".
-- 2. create temporary table tst(x int);
-- 3. The results pane in the bottom will have a blue button stating "Go to this table". Click it.
-- 4. This will open the temporary session (which lasts for 24 hrs...can be seen from expires at timestamp) that BQ has created for this tab. Copy the temporary dataset_id (ought to begin with the prefix '_') and prefix calendar_year with it. You will have to use bquotes!
--    4a. create or replace table function `«project_id»._«temporary-session-id»`.calendar_year
--    4b. Annoyingly, i cannot call this TVF (table-valued function) without the complete path!
create or replace table function calendar_year(yr any type) as (with _ as (select cast(dt as date) as dt from unnest(generate_date_array(cast(yr||'-01-01' as date),cast(yr||'-12-31' as date),interval 1 day)) as dt) select dt,format_date('%a',dt) as dow,cast(format_date('%Y%Q',dt) as int) as yyyyqtr from _);


create or replace function lastarrayelement(arr any type) as ( arr[safe_ordinal(array_length(arr))]);
-- select lastarrayelement(a) from (select [1,2,3,4] as a union all select [] union all select [1]) s;
-- format-bq-query removes lines beginning with assert! Therefore, please keep assert[s] on one line!
assert lastarrayelement([1,2,3,4]) = 4;
assert lastarrayelement([]) is null;
assert lastarrayelement([1]) = 1;
assert lastarrayelement([struct(1 as x, 2 as y),(11,12),(21,22),(31,32)]) = (31,32);
-- select lastarrayelement(15) should raise error! Since SQL is strongly typed arr can be only ARRAY type! Therefore BQ will raise an error here! SQL is awesome!

create temp function are_arrays_equal(arr1 any type, arr2 any type) as (
  if(array_length(arr1) <> array_length(arr2), false,
    if(array_length(arr1)=0,true,
      (select logical_and(x=arr2[safe_offset(idx)]) from unnest(arr1) as x with offset as idx)))
);
assert (select logical_and(are_arrays_equal(n.f1,n.f2)=n.f3) from unnest([struct([] as f1, [] as f2, true as f3),([1],[],false),([1,2],[1,2],true),([],[1],false)]) as n) = true;
assert are_arrays_equal([1],[]) = false;
assert are_arrays_equal([],[]) = true;
assert are_arrays_equal([1,2,3],[1,2,3]) = true;
assert are_arrays_equal([1,2,3],[1.0,2,3]) = true; -- Arrays promoted to supertype double yielding ARRAY<DOUBLE> before comparison!
-- assert are_arrays_equal([1,2,3],['a','b','c']) -- This will be unallowed by the typechecker!!! BQ is awesome!
-- The above function does not work when comparing array of struct which contains arrays. This is because BQ does not allow comparing arrays. This affects comparing structs too!
-- with vars as (select struct(1 as id, 'item1' as lbl) as v1, struct(1.0 as tid, 'item1' as tt) as v2) select *,v1=v2 are_equal from vars; -- this works
-- with vars as (select struct([1] as id, 'item1' as lbl) as v1, struct([1.0] as tid, 'item1' as tt) as v2) select *,v1=v2 are_eq from vars; -- this will NOT work! Because while structs are comparable when the constituent components are not ARRAY.
-- Unsure if these restrictions apply for GEOGRAPHY (doubtful) and JSON (also doubtful). ## TODO (vijay): Verify this...

create temp function array_contains(arr any type, elt any type) as (
  if(array_length(arr)=0,false,
   -- (select logical_or(elt=x) from unnest(arr) as x)
   (EXISTS(select * from unnest(arr) as x where x = elt))
  )
);
assert array_contains([1,2,3],1) = true;
assert array_contains([],4) = false;
assert array_contains([1,2,3],1.0) = true;
assert array_contains([1,2,3],1.5) = false;
-- Again BQ maintains type conformity!! array_contains([1,2,3],'vl') is a type error!

create temp function array_unique(arr any type) as (
  (select coalesce(array_agg(distinct n), []) from unnest(arr) as n) -- coalesce needed to deal with empty arrays not yielding any rows which makes array_agg return null!
);
assert array_length(array_unique([])) = array_length([]);
assert (select logical_and(array_contains([1,2,3,4,5,6],n)) from unnest(array_unique([1,2,3,1,2,3,4,5,6])) as n) and (array_length(array_unique([1,2,3,1,2,3,4,5,6]))=6);
assert (select logical_and(array_contains([4,1,3,5,6,2],n)) from unnest(array_unique([1,2,3,1,2,3,4,5,6])) as n);

create or replace temp function array_proportion(arr any type) as (
 -- similar to R's prop.table or proportions function
 -- ignores null
 (
   with _ as (select a,aidx from unnest(arr) as a with offset aidx where a is not null) -- NOTE (vijay): bq does not allow null in array ... redundant?
   , _1 as (select a,aidx,a/sum(a) over() as pct from _)
   select coalesce(array_agg(struct(a,pct) order by aidx),[]) from _1
 )
);
-- TODO (vijay): figure out how to check these... i.e., how to compare structs...
-- assert are_arrays_equal(array_proportion([]),[]);
-- assert are_arrays_equal(array_proportion([4,4,4,4]),[(4,0.25),(4,0.25),(4,0.25),(4,0.25)]);
-- assert are_arrays_equal(array_proportion([0.04514891774626828,0.4126947136518826,0.7758699677504728]),[(0.04514891774626828,0.03659594720964834),(0.4126947136518826,0.3345141967606421),(0.7758699677504728,0.6288898560297096)]);


-- some debugging related ideas...from the BQ book!
-- create temp function debugarray(arr any type) as (array_to_string(arr, '*', '«»'));
-- assert debugarray(['A','B',null,'D']) = 'A*B*«»*D';
create temp function debugvalue(v any type) as (to_json_string(v)); -- to_json_string is much more general and works with everything...not just arrays!
assert debugvalue(['A','B',null,'D']) = '["A","B",null,"D"]';
assert debugvalue([struct('A' as a,'B' as b,null as c,'D' as d)]) = '[{"a":"A","b":"B","c":null,"d":"D"}]';

-- dot product
-- NOTE (vijay) 2024.05.01: Apparently `dot_product` appears to be reserved built-in name for future use by BQ! Therefore, renaming this function to ddot_product.
CREATE TEMP FUNCTION ddot_product(a1 ANY type, a2 ANY type) AS (
    CASE WHEN ARRAY_LENGTH(a1) <> ARRAY_LENGTH(a2) THEN ERROR('Lengths ' || array_length(a1) || ' and ' || array_length(a2) || ' are not equal!')
    ELSE
    (
      WITH
	tbl1 AS ( SELECT n1, idx FROM UNNEST(a1) AS n1 WITH OFFSET AS idx),
	tbl2 AS ( SELECT n2, idx FROM UNNEST(a2) AS n2 WITH OFFSET AS idx)
      SELECT SUM(n1*n2) FROM tbl1 INNER JOIN tbl2 USING (idx)
    )
    END
);
ASSERT ddot_product(ARRAY[1, 2, 5], ARRAY[1, 2, 3]) = 20;
-- ASSERT ddot_product(ARRAY[1, 2, 5], ARRAY[1, 2, 3, 4]) = 'Lengths 3 and 4 are not equal!'; -- TODO (vijay): figure out how to check this....

CREATE TEMP FUNCTION magnitude(arr any type) AS (
  sqrt(ddot_product(arr, arr))
);
ASSERT magnitude(ARRAY[1,2,5]) = sqrt(30);
ASSERT magnitude(ARRAY[1,2,3]) = sqrt(14);

-- BQ has cosine_distance which is 1 - cosine_similarity! Better to use cosine_distance than rely on this function here!
CREATE TEMP FUNCTION cosine_similarity(a1 ANY type, a2 ANY type) AS (
  ddot_product(a1, a2)/(magnitude(a1)*magnitude(a2))
);
ASSERT cosine_similarity(ARRAY[0,0,0,1,1,1,1,1,2,1,2,0,1,0], ARRAY[0,1,1,1,0,0,1,0,1,1,1,0,1,1]) = 1 - cosine_distance(ARRAY[0.0,0,0,1,1,1,1,1,2,1,2,0,1,0], ARRAY[0.0,1,1,1,0,0,1,0,1,1,1,0,1,1]);
ASSERT cosine_similarity(ARRAY[0,1,1,1,0,0,1,0,1,1,1,0,1,1], ARRAY[1,0,0,2,0,0,0,0,0,0,0,1,0,0]) = 1 - cosine_distance(ARRAY[0.0,1,1,1,0,0,1,0,1,1,1,0,1,1], ARRAY[1.0,0,0,2,0,0,0,0,0,0,0,1,0,0]);
-- ASSERT cosine_similarity(ARRAY[0,0,0,1,1,1,1,1,2,1,2,0,1,0], ARRAY[1,0,0,2,0,0,0,0,0,0,0,1,0,0]) = 1 - cosine_distance(ARRAY[0.0,0,0,1,1,1,1,1,2,1,2,0,1,0], ARRAY[1.0,0,0,2,0,0,0,0,0,0,0,1,0,0]);
ASSERT cast(cosine_similarity(ARRAY[0,0,0,1,1,1,1,1,2,1,2,0,1,0], ARRAY[1,0,0,2,0,0,0,0,0,0,0,1,0,0]) as numeric) = cast(1 - cosine_distance(ARRAY[0.0,0,0,1,1,1,1,1,2,1,2,0,1,0], ARRAY[1.0,0,0,2,0,0,0,0,0,0,0,1,0,0]) as numeric);


-- Like Python's divmod function...i've needed this more than a few times...
create temp function divmod(x any type, y any type) as ( (div(x,y), mod(x,y)) );
assert divmod(199001,100) = (1990, 1); -- Especially useful if you store YYYYMM as an int!!

create temp aggregate function positivesum(x numeric) as ( if(sum(x)<0, null, sum(x)) ); -- constrain aggregate sum to [0, inf). Useful for revenue and counts which are required to +ve only. NOTE: temporary aggregate functions are disallowed in PIVOT clause!

create or replace function money_to_numeric(m any type) as (cast(replace(replace(replace(replace(d,'$',''),',',''),')',''),'(','-') as numeric) );
assert money_to_numeric('(1,283.4)') = -1283.4;
assert money_to_numeric('$(1,283.4)') = -1283.4;
assert money_to_numeric('(1,234,567.8') = -1234567.8;
assert money_to_numeric(' 1,234,567.8') =  1234567.8; -- spaces handled automatically? NOTE (vijay): verify this...

- TODO (vijay): figure out how to make these work with RANGE type
create or replace function squote(s any type) as (format("'%t'",s));
create or replace function dquote(s any type) as (format('"%t"',s));

-- geospatial realated stuff
create or replace function deg2rad(deg any type) as ( deg * acos(-1) / 180 ); assert deg2rad(180) = acos(-1); -- acos(-1) is pi!
create or replace function rad2deg(rad any type) as ( rad * 180 / acos(-1) ); assert rad2deg(acos(-1)) = 180;
create or replace function cart2pol(x any type, y any type) as ( struct( sqrt(x*x + y*y) as rho, atan2(y, x) as phi) ); assert cart2pol(12, 5) = (13, 0.3947911); -- phi is in radians!
create or replace function pol2cart(rho any type, phi any type) as ( struct(rho*cos(phi) as x, rho*sin(phi) as y) ); assert pol2cart(13, 0.3947911) = (12, 5);
