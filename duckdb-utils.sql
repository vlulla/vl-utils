-- bash $ duckdb
-- D .read 'utils.sql'
-- D

-- select yqtr,count(*),min(dt) first_dt,max(dt) as last_dt from (select dt,yearqtr(dt) as yqtr from (select unnest(generate_series('2024-01-01'::date,'2024-12-31',interval 1 day))::date as dt)) group by all order by yqtr;
create macro yearqtr(x) as (select (date_part('year', x::date)*10+((date_part('month', x::date)-1) // 3) + 1));
-- select * from calendar_year(2024);
create macro calendar_year(yr) as TABLE (with _ as (select DT::DATE as DT from unnest(generate_series(cast(YR || '-01-01' as date), cast(YR || '-12-31' as date), interval '1' day)::date[]) as _(dt)) select dT, strftime(dt, '%a') as dow, yearqtr(dt) as YYYYQTR from _);
