#!/usr/bin/env -S sed -E -f
## Call this script like
## sed -E -f format-bq-query.sed tst.sql

## Need this sed script to remove all the extraneous stuff from the sql query so that BQ can cache the query!
##
## BQ caching is dependent upon the string so any change (even adding/deleting spaces/commments and/or changing case) will
## make BQ behave as it is a different query! A better solution might be to run the query through this filter and normalize
## the sql script so that we can benefit, hopefully (don't know how to check this though), from BQ!
##
##
## 2023.04.04:
## bash $ sed -E -f format-bq-query.sed tst.sql | pbcopy
## bash $ sed -E -f format-bq-query.sed tst.sql | vim -
##
## Now you can paste the output into the BQ console window


/^[[:space:]]*--/d
/^assert /d

s/[[:space:]]--.*$//g
s/[[:space:]]+/ /g
/^[[:space:]]*$/d

s/` /`/g
s/([^'])`.`([^'])/\1.\2/g
