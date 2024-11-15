-- bash $ duckdb
-- D .read 'utils.sql'
-- D

-- select yqtr,count(*),min(dt) first_dt,max(dt) as last_dt from (select dt,yearqtr(dt) as yqtr from (select unnest(generate_series('2024-01-01'::date,'2024-12-31',interval 1 day))::date as dt)) group by all order by yqtr;
create macro yearqtr(x) as (select (date_part('year', x::date)*10+((date_part('month', x::date)-1) // 3) + 1));
-- select * from calendar_between_dates('2024-06-15','2025-02-13'); -- calendar dates between custom dates!!
create macro calendar_between_dates(start_date,end_date) as table (with _ as (select unnest(generate_series(start_date::date,end_date::date,interval 1 day)::date[])::date as dt) select dt,strftime(dt,'%a') as dow, yearqtr(dt) as YYYYQTR from _);
-- select * from calendar_year(2024);
create macro calendar_year(yr) as TABLE (select * from calendar_between_dates(yr||'-01-01', yr||'-12-31'));
