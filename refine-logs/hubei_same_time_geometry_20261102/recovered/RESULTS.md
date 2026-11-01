# 湖北修正太阳几何：真实三帧同期GHI验证

{
  "state": "COMPLETE",
  "scope": "Hubei corrected solar geometry; observed T-30/T-15/T frames; paired validation",
  "station_seed_equal_RMSE": {
    "A": 74.4056169636632,
    "AC": 68.62918776427138
  },
  "relative_RMSE_reduction_pct": 7.7634316266913155,
  "delta_AC_minus_A": -5.776429199391828,
  "date_block_95CI": [
    -9.650115636617192,
    -2.3129539902040976
  ],
  "unique_validation_station_times": 1905,
  "by_station": {
    "Niushou": {
      "A_rmse": 69.96791627585837,
      "AC_rmse": 64.4511686482329
    },
    "Wangzhai": {
      "A_rmse": 67.45190675951436,
      "AC_rmse": 59.214625315926526
    },
    "Mizhuang": {
      "A_rmse": 85.79702785561686,
      "AC_rmse": 82.22176932865467
    }
  },
  "test_payloads_read": 0
}

日期块配对区间为5000次目标BJT日期重采样。逐站、逐种子先算RMSE后等权，不混池。非独立test。
