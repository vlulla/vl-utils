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

create or replace function lastarrayelement(arr any type) as ( arr[safe_ordinal(array_length(arr))]);
select lastarrayelement(a) from (select [1,2,3,4] as a union all select [] union all select [1]) s;
-- format-bq-query removes lines beginning with assert! Therefore, please keep assert[s] on one line!
assert lastarrayelement([1,2,3,4]) = 4;
assert lastarrayelement([]) is null;
assert lastarrayelement([1]) = 1;
assert lastarrayelement([struct(1 as x, 2 as y),(11,12),(21,22),(31,32)]) = (31,32);
-- select lastarrayelement(15) should raise error! Since SQL is strongly typed arr can be only ARRAY type! Therefore BQ will raise an error here! SQL is awesome!

create or replace function deg2rad(deg any type) as ( deg * acos(-1) / 180 ); assert deg2rad(180) == acos(-1); -- acos(-1) is pi!
create or replace function rad2deg(rad any type) as ( rad * 180 / acos(-1) ); assert rad2deg(acos(-1)) == 180;
create or replace function cart2pol(x any type, y any type) as ( struct( sqrt(x*x + y*y) as rho, atan2(y, x) as phi) ); assert cart2pol(12, 5) = (13, 0.3947911); -- phi is in radians!
create or replace function pol2cart(rho any type, phi any type) as ( struct(rho*cos(phi) as x, rho*sin(phi) as y) ); assert pol2cart(13, 0.3947911) = (12, 5);
