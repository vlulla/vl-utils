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
create or replace macro array_prop(d) as (with _ as (select list_reduce(list_filter(d,lambda _:_ is not null),lambda a,b:a+b) as dsum) select list_transform(d,lambda x:x/dsum) from _);

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

-- holiday features for various timeseries forecasting related tasks...
create or replace macro datefeatures(startdate, enddate) as TABLE
(
with
  dates as (select unnest(generate_series(cast(startdate as date),cast(enddate as date),interval 1 day)::date[]) as date)
, dates_with_attrs as (select  date,extract(year from date) as year, extract(month from date) as month, extract(day from date) as day, strftime(date, '%a') as dow,strftime(date,'%j')::int64 as doy from dates)
, dates_with_attrs_with_ordinals as (select *,row_number() OVER (yearmonweekday order by date) as ordinal_dow from dates_with_attrs window yearmonweekday as (partition by year,month,dow))
, newyears as (select date,cast((month,day)=(1,1) as int) as is_newyears_day from dates_with_attrs)
, mlkdays as (select date,cast((month,dow,ordinal_dow)=(1,'Mon',3) and year>1982 as int) as is_mlkjr_day from dates_with_attrs_with_ordinals)
, valentines as (select date,cast((month,day)=(2,14) as int) as is_valentines_day from dates_with_attrs)
, memorialdays as (select date,cast(date in (select max(date) from dates_with_attrs_with_ordinals where (month,dow)=(5,'Mon') group by year) as int) as is_memorial_day from dates)
, mothersdays as (select date,cast((month,dow,ordinal_dow)=(5,'Sun',2) as int) as is_mothersday from dates_with_attrs_with_ordinals)
, fathersdays as (select date,cast((month,dow,ordinal_dow)=(6,'Sun',3) as int) as is_fathersday from dates_with_attrs_with_ordinals)
, juneteenthdays as (select date,cast((month,day)=(6,19) and year>=2021 as int) as is_juneteenth_day from dates_with_attrs)
, july4days as (select date,cast((month,day)=(7,4) as int) as is_july4_day from dates_with_attrs)
, labordays as (select date,cast((month,dow,ordinal_dow)=(9,'Mon',1) as int) as is_labor_day from dates_with_attrs_with_ordinals)
, thanksgivingdays as (select date,cast((month,dow,ordinal_dow)=(11,'Thu',4) as int) as is_thanksgiving_day from dates_with_attrs_with_ordinals)
, blackfridays as (select date,cast(date in (select date (date+interval 1 day) from thanksgivingdays where is_thanksgiving_day<>0) as int) as is_blackfriday from dates)
, cybermondays as (select date,cast(date in (select date (date+interval 4 day) from thanksgivingdays where is_thanksgiving_day<>0) as int) as is_cybermonday from dates)
, christmasdays as (select date,cast((month,day)=(12,25) as int) as is_christmas_day from dates_with_attrs)
, weekdayfeatures as (select date, cast(dow='Mon' as int) as is_monday, cast(dow='Tue' as int) as is_tuesday, cast(dow='Wed' as int) as is_wednesday, cast(dow='Thu' as int) as is_thursday, cast(dow='Fri' as int) as is_friday, cast(dow='Sat' as int) as is_saturday, cast(dow='Sun' as int) as is_sunday from dates_with_attrs)
, monthfeatures as (select date,cast(month=1 as int) as is_jan, cast(month=2 as int) as is_feb, cast(month=3 as int) as is_mar, cast(month=4 as int) as is_apr, cast(month=5 as int) as is_may, cast(month=6 as int) as is_jun, cast(month=7 as int) as is_jul, cast(month=8 as int) as is_aug, cast(month=9 as int) as is_sep, cast(month=10 as int) as is_oct, cast(month=11 as int) as is_nov, cast(month=12 as int) as is_dec from dates_with_attrs)
, weekendfeatures as (select date,(is_monday|is_tuesday|is_wednesday|is_thursday|is_friday) as is_weekday,(is_saturday|is_sunday) as is_weekend from weekdayfeatures)
select * from
          -- dates
          dates_with_attrs
          -- dates_with_attrs_with_ordinals
left join newyears         using(date)
left join mlkdays          using(date)
left join valentines       using(date)
left join mothersdays      using(date)
left join memorialdays     using(date)
left join fathersdays      using(date)
left join juneteenthdays   using(date)
left join july4days        using(date)
left join labordays        using(date)
left join thanksgivingdays using(date)
left join blackfridays     using(date)
left join cybermondays     using(date)
left join christmasdays    using(date)
left join weekdayfeatures  using(date)
left join monthfeatures    using(date)
left join weekendfeatures  using(date)
where 1=1
-- and year(date) between 2020 and 2030
-- and is_newyears_day=1
-- and is_mlkjr_day=1
-- and is_valentines_day=1
-- and is_mothersday_day=1
-- and is_memorial_day=1
-- and is_fathersday_day=1
-- and is_juneteenth_day=1
-- and is_july4_day=1
-- and is_labor_day=1
-- and is_thanksgiving_day=1
-- and is_blackfriday_day=1
-- and is_cybermonday_day=1
-- and is_christmas_day=1
-- and is_weekend=0
order by date
-- limit 10
);
-- select * from datefeatures('2025-01-01','2025-12-31');

-- workaround for missing difference operator for time data type
create or replace macro time_diff(t1, t2) as ('2000-01-01T'||t1)::timestamp - ('2000-01-01T'||t2)::timestamp;
-- select time_diff(time '20:18:32.05', time '18:00:28.1'); -- same as time '20:18:32.05' - time '18:00:28.1'

-- D VACUUM ANALYZE;
-- D call table_sizes(); -- to get dims of table...
create or replace macro table_sizes() as TABLE (select database_name,schema_name,table_name,estimated_size as nrow,column_count as ncol from duckdb_tables());
