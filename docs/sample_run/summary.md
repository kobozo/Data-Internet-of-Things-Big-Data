# Analysis summary

* observation window : **2026-05-27 13:01:15.906426+00:00 -> 2026-05-27 13:01:54.906426+00:00**
* frames processed   : **40**
* avg people / frame : **3.35**
* overall tourist share: **80.6%**

* tracks observed    : **5**
* breakdown by class :
  * `TOURIST` : 4
  * `LOCAL` : 1

## Events fired
* `tourist_hotspot` x 1
* `tourist_group_arrived` x 1

## Big-data context

A single street camera at 2 fps produces ~170 k frame-rows per day.
Scaling to 200 cameras across a city = ~34 M rows/day, which is the
natural break-even point where SQLite stops working and a real
OLAP store (ClickHouse, BigQuery, InfluxDB) is required.  The
schema we already write (`ts`, `n_people`, `n_tourists`,
`avg_dwell`) lands directly into a time-series database without
transformation; events go into a separate, lower-volume topic that
drives the alerting layer.
