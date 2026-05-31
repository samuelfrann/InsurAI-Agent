export const VEHICLE_PRICE_BINS = [
  { label: 'Less than ₦32,000,000',         usdMax: 20000,  bin: 'less than 20000' },
  { label: '₦32,000,000 – ₦46,400,000',     usdMax: 29000,  bin: '20000 to 29000' },
  { label: '₦48,000,000 – ₦62,400,000',     usdMax: 39000,  bin: '30000 to 39000' },
  { label: '₦64,000,000 – ₦94,400,000',     usdMax: 59000,  bin: '40000 to 59000' },
  { label: '₦96,000,000 – ₦110,400,000',    usdMax: 69000,  bin: '60000 to 69000' },
  { label: 'More than ₦110,400,000',         usdMax: Infinity, bin: 'more than 69000' },
]

export function nairaToUsd(naira, rate) {
  return naira / rate
}

export function usdToVehiclePriceBin(usd) {
  for (const tier of VEHICLE_PRICE_BINS) {
    if (usd <= tier.usdMax) return tier.bin
  }
  return 'more than 69000'
}

export function nairaToDeductibleBin(naira, rate) {
  const usd = nairaToUsd(naira, rate)
  const valid = [300, 400, 500, 700]
  return valid.reduce((prev, curr) =>
    Math.abs(curr - usd) < Math.abs(prev - usd) ? curr : prev
  )
}