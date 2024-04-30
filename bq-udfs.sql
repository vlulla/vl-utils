-- Some functions I find useful in sql explorations.
--
-- See
-- UserGuide: https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions
-- Reference: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_function_statement

create or replace function round2(x any type) as ( round(x, 2) );
create or replace function round3(x any type) as ( round(x, 3) );

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



-- some debugging related ideas...from the BQ book!
-- create temp function debugarray(arr any type) as (array_to_string(arr, '*', '«»'));
-- assert debugarray(['A','B',null,'D']) = 'A*B*«»*D';
create temp function debugvalue(v any type) as (to_json_string(v)); -- to_json_string is much more general and works with everything...not just arrays!
assert debugvalue(['A','B',null,'D']) = '["A","B",null,"D"]';
assert debugvalue([struct('A' as a,'B' as b,null as c,'D' as d)]) = '[{"a":"A","b":"B","c":null,"d":"D"}]';


-- Like Python's divmod function...i've needed this more than a few times...
create temp function divmod(x, y) as ( (div(x,y), mod(x,y)) );
assert divmod(199001,100) = (1990, 1); -- Especially useful if you store YYYYMM as an int!!

-- geospatial realated stuff
create or replace function deg2rad(deg any type) as ( deg * acos(-1) / 180 ); assert deg2rad(180) == acos(-1); -- acos(-1) is pi!
create or replace function rad2deg(rad any type) as ( rad * 180 / acos(-1) ); assert rad2deg(acos(-1)) == 180;
create or replace function cart2pol(x any type, y any type) as ( struct( sqrt(x*x + y*y) as rho, atan2(y, x) as phi) ); assert cart2pol(12, 5) = (13, 0.3947911); -- phi is in radians!
create or replace function pol2cart(rho any type, phi any type) as ( struct(rho*cos(phi) as x, rho*sin(phi) as y) ); assert pol2cart(13, 0.3947911) = (12, 5);
