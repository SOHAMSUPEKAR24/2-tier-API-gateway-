-- load_test/lua/escalated.lua
-- Pure Escalated Traffic: 100% requests fall into the uncertainty band and undergo Tier 2 deep verification

local counter = 0
local esc_body = "modo=registro&login=glaser&password=elementada&nombre=Davel&apellidos=De+Mendonca+Ors&email=chua%40comerciosdeaspe.om&dni=41909752A&direccion=De+La+Coma%2C+112+4-B&ciudad=San+Miguel+de+Abona&cp=04100&provincia=Ourense&ntc=4150902070293588&B1=Registrar"

request = function()
    counter = counter + 1
    local headers = {}
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    headers["User-Agent"] = "Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)"
    headers["Accept"] = "text/html,application/xhtml+xml,text/xml;q=0.9,*/*;q=0.8"
    headers["Host"] = "127.0.0.1:8080"
    return wrk.format("POST", "/tienda1/miembros/editar.jsp", headers, esc_body)
end

done = function(summary, latency, requests)
    io.write(string.format([[JSON_METRICS:{"p50_us": %f, "p90_us": %f, "p95_us": %f, "p99_us": %f, "mean_us": %f, "requests": %d, "duration_us": %d, "rps": %f}]],
        latency:percentile(50.0),
        latency:percentile(90.0),
        latency:percentile(95.0),
        latency:percentile(99.0),
        latency.mean,
        summary.requests,
        summary.duration,
        (summary.requests / (summary.duration / 1000000.0))
    ))
end
