-- Some functions I find useful in sql explorations.
--
-- See
-- UserGuide: https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions
-- Reference: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_function_statement

create or replace function round2(x any type) as ( round(x, 2) );
create or replace function round3(x any type) as ( round(x, 3) );

create or replace function iso_yyyyweek(x any type) as ( cast(format_date("%G%V", cast(x as date) ) as int64) );
create or replace function iso_yyyyday (x any type) as ( cast(format_date("%G%J", cast(x as date) ) as int64) ); -- 001-364 OR 001-371  !?!?!
assert iso_yyyyday('2023-01-01') = 2022364; assert iso_yyyyday('2023-01-02') = 2023001;

create or replace function deg2rad(deg any type) as ( deg * acos(-1) / 180 ); assert deg2rad(180) == acos(-1); -- acos(-1) is pi!
create or replace function rad2deg(rad any type) as ( rad * 180 / acos(-1) ); assert rad2deg(acos(-1)) == 180;
create or replace function cart2pol(x any type, y any type) as ( struct( sqrt(x*x + y*y) as rho, atan2(y, x) as phi) ); assert cart2pol(12, 5) = (13, 0.3947911); -- phi is in radians!
create or replace function pol2cart(rho any type, phi any type) as ( struct(rho*cos(phi) as x, rho*sin(phi) as y) ); assert pol2cart(13, 0.3947911) = (12, 5);
