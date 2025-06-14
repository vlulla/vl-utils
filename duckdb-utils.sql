-- bash $ duckdb
-- D .read 'utils.sql'
-- D

-- select yqtr,count(*),min(dt) first_dt,max(dt) as last_dt from (select dt,yearqtr(dt) as yqtr from (select unnest(generate_series('2024-01-01'::date,'2024-12-31',interval 1 day))::date as dt)) group by all order by yqtr;
-- create or replace macro yearqtr(x) as (select (date_part('year', x::date)*10+((date_part('month', x::date)-1) // 3) + 1));
create or replace macro yearqtr(x) as (select (year(x::date)*10)+quarter(x::date));

-- select * from calendar_between_dates('2024-06-15','2025-02-13'); -- calendar dates between custom dates!!
create or replace macro calendar_between_dates(start_date,end_date) as table (with _ as (select unnest(generate_series(start_date::date,end_date::date,interval 1 day)::date[])::date as dt) select dt,strftime(dt,'%a') as dow, year(dt) as YYYY, month(dt) as M, dayofmonth(dt) DOM, weekofyear(dt) as WOY, yearweek(dt) as ISOYYYYWK,isodow(dt) as ISODOW,dayofweek(dt) as dow_num,quarter(dt) as Q, YYYY||''||Q as YYYYQ from _);

-- select * from calendar_year(2024);
create or replace macro calendar_year(yr) as TABLE (select * from calendar_between_dates(yr||'-01-01', yr||'-12-31'));

-- select * from random_date_range(10);
-- select * from random_date_range(floor(365*random())::int); -- even better!
-- with _ as (select * from random_date_range(floor(365*random())::int)) select length(dates) as len,dates from _;
-- with _ as (select floor(random()*50)::int as n) select n,dates from _ JOIN random_date_range(n) on TRUE; -- n comes from _ CTE!
-- with _ as (select floor(random()*50)::int as n) select n,dates from _,random_date_range(n); -- same as previous...but JOIN is implicit...
create or replace macro random_date_range(ndays) as TABLE (with _ as (select '2000-01-01'::date + floor((current_date() - '2000-01-01'::date)*random())::int as start_date) select generate_series(start_date,start_date + ndays - 1,interval 1 day)::date[] as dates from _);

-- select * from days_around_date('2024-11-28'::date,5); -- query to generate a window around a date. This is very useful for creating windows to compare times from different years.
-- Here's an interesting use case: comparing a month around thanksgiving/black friday for different years. For instance, comparing cybermondays on 2023 and 2024 can be compared by using date axes: days_around_date('2024-12-02'::date,15) and days_around_date('2023-11-27'::date,15)!!
-- select unnest(dates) as dt from days_around_date('2024-12-02'::date,5);
-- with yr2024 as (select unnest(range(length(dates))) as rn,unnest(dates) as dt2024 from days_around_date('2024-12-02'::date,15))
--     ,yr2023 as (select unnest(range(length(dates))) as rn,unnest(dates) as dt2023 from days_around_date('2023-11-27'::date,15))
-- select * exclude(rn) from yr2023 join yr2024 using(rn);
create or replace macro days_around_date(date,ndays) as TABLE(select generate_series(date-interval (ndays) day,date-interval 1 day,interval 1 day)::date[]||[date]||generate_series(date+interval 1 day,date+interval (ndays) day,interval 1 day)::date[] as dates);

-- This date_trunc is analogous to BQ's default date_trunc with default week date_granularity. BQ's date_trunc is interesting in that you can select what represents the first day of the week (dow).
-- date_trunc(current_date(), week) -- default date_granularity ... same as date_trunc(current_date(), week(sunday))
-- date_trunc(current_date(), week(MONDAY)) -- This is the ISO-8601 week definition...first dow is Monday!
--
-- However, for certain situations it might be useful to consider a week from Sun-Sat instead of Mon-Sun. Since, Sun-Sat week grouping comes up so often that i think creating this macro is useful. Use it along with duckdb's default `date_trunc` function to get different aggregations...
--
-- with _ as (select unnest(generate_series(current_date()-interval 5 day,current_date()+interval 5 day,interval 1 day))::date as _) select _,date_trunc('week',_) as defaut_date_trunc,date_trunc_week_like_bq(_) as "date_trunc-like-BQ" from _;
create or replace macro date_trunc_week_like_bq(dt) as (select (date_trunc('week',dt::date+interval 1 day)-interval 1 day)::date);

-- convert currency formatted numbers into mathematical numbers
-- can only deal with '$'!
create or replace macro money_to_numeric(d) as ( select replace(replace(replace(replace(d,'$',''),',',''),')',''),'(','-')::numeric(18,2) );
-- with _ as (select unnest(['-$1,234.10','$1,234.10','($1,234.10)','($0.99999)', '   $0.3333333333333333', '    ($0.3333333333333333)     ', '$0.0', '($6.080089694185882e-110)','($4.080089694185882e-10)     ']) as m) select m,printf('"%s"',m) as "m with quotes",money_to_numeric(m) as mm from _;

-- Proportions in the array...maintains nulls!
create or replace macro array_prop(d) as (with _ as (select list_reduce(list_filter(d,lambda _:_ is not null),lambda a,b:a+b) as dsum) select list_transform(d,x->x/dsum) from _);

-- random str of length len
create or replace macro randomstr(len) as (
  -- with _ as (select 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' as alpha,'0123456789' as num)
  --   , _1 as (select *,alpha||num as alnum,length(alpha) as alen,length(num) as nlen from _)
  -- --   , _2 as (select alnum,cast(trunc(random()*alen) as int) as fst, apply(range(1,len),lambda x:cast(trunc(random()*(alen+nlen))+1 as int)) as rest from _1)
  -- -- select list_reduce(apply([fst]||rest,lambda i:alnum[i]),lambda l,r:l||r) as str from _2
  -- , _2 as (select alnum,apply(range(len),lambda x:cast(trunc(random()*(alen+nlen))+1 as int)) as idx from _1)
  -- select case when length(idx)=0 then '' else list_reduce(apply(idx,lambda i:alnum[i]),lambda l,r:l||r) end as str from _2
  select string_agg(alnum[idx],'') as randstr from (
    select 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' as alnum,cast(trunc(random()*length(alnum))+1 as int) as idx from range(len)
  )
);
-- select randomstr(cast(100*random() as int));
