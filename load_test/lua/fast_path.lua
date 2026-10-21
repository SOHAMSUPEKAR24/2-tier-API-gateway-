-- load_test/lua/fast_path.lua
-- Pure Benign Fast-Path Traffic: 100% requests screened with low-latency edge inspection by Tier 1

local counter = 0
local paths = {
    "/tienda1/global/creditos.jsp",
    "/tienda1/publico/carrito.jsp",
    "/tienda1/publico/miembros.jsp",
    "/tienda1/publico/productos.jsp"
}

request = function()
    counter = counter + 1
    local p = paths[(counter % #paths) + 1]
    local headers = {}
    headers["User-Agent"] = "Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)"
    headers["Accept"] = "text/html,application/xhtml+xml,text/xml;q=0.9,*/*;q=0.8"
    headers["Host"] = "127.0.0.1:8080"
    return wrk.format("GET", p, headers, nil)
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
