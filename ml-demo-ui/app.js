const API_URL = "https://soltrice-sfm-api.onrender.com/score";

const HEALTH_URL = API_URL.replace(/\/score$/, "/health");
fetch(HEALTH_URL, { cache: "no-store" }).catch(() => {});


/* =======================
   Helpers
======================= */

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

/* ---- Local fallback scoring (kept as backup) ---- */
function calculateRiskScore(inputs) {
  let score = 0;

  // Customer signals
  score += (30 - Math.min(inputs.emailAge, 30)) * 0.7;        // young email
  score += (20 - Math.min(inputs.tenure / 6, 20)) * 0.6;      // short tenure
  score += inputs.chargebacks * 1.6;                          // disputes
  score -= inputs.premiumFlag * 10;                           // premium discount

  // Driver
  if (inputs.driverAge < 23) score += (23 - inputs.driverAge) * 1.5;
  else if (inputs.driverAge > 70) score += (inputs.driverAge - 70) * 0.7;
  else score -= 2;

  if (inputs.licenseYears < 5) score += (5 - inputs.licenseYears) * 1.5;
  else if (inputs.licenseYears > 10) score -= (Math.min(inputs.licenseYears, 25) - 10) * 0.4;

  score -= Math.min(inputs.previousRentals, 20) * 0.5;

  if (inputs.phoneType === 2) score += 8;        // VoIP
  else if (inputs.phoneType === 0) score -= 4;   // landline/office

  // Transaction
  score += (inputs.amount / 50) * 0.9;
  score += inputs.velocity * 0.9;
  score += (inputs.binRisk / 10) * 1.2;
  score += inputs.oddHour ? 6 : 0;

  // Device & network
  score += (30 - Math.min(inputs.deviceTrust / 3.3, 30)) * 1.2;
  score += (inputs.ipRisk / 8);
  score += (inputs.geoDistance / 150) * 1.2;
  score += inputs.proxyFlag ? 10 : 0;

  score = clamp(score, 0, 100);
  return Math.round(score);
}

/* =======================
   Sphere / Tier / Insights (unchanged)
======================= */

function determineRiskTier(score) {
  if (score <= 10) {
    return {
      label: "Premium Customer · Very Low Risk",
      assessment: "Near-zero fraud risk based on current signals.",
      sphereColor: "#22c55e",
      glow: "rgba(34, 197, 94, 0.75)",
      badge: "Premium · Green",
      textClass: "text-positive"
    };
  } else if (score <= 25) {
    return {
      label: "Very Low Risk",
      assessment: "Signals are strongly in favor of this customer.",
      sphereColor: "#4ade80",
      glow: "rgba(74, 222, 128, 0.7)",
      badge: "Very Low Risk",
      textClass: "text-positive"
    };
  } else if (score <= 45) {
    return {
      label: "Low to Moderate",
      assessment: "Mostly safe, but one or two dimensions need watching.",
      sphereColor: "#eab308",
      glow: "rgba(234, 179, 8, 0.75)",
      badge: "Moderate Risk",
      textClass: ""
    };
  } else if (score <= 65) {
    return {
      label: "Elevated Risk",
      assessment: "Requires additional checks before approving.",
      sphereColor: "#f97316",
      glow: "rgba(249, 115, 22, 0.78)",
      badge: "Elevated Risk",
      textClass: "text-negative"
    };
  } else if (score <= 80) {
    return {
      label: "High Risk",
      assessment: "Multiple strong fraud indicators present.",
      sphereColor: "#ef4444",
      glow: "rgba(239, 68, 68, 0.82)",
      badge: "High Risk",
      textClass: "text-critical"
    };
  } else {
    return {
      label: "Critical Risk",
      assessment: "Block or send to manual review immediately.",
      sphereColor: "#b91c1c",
      glow: "rgba(185, 28, 28, 0.88)",
      badge: "Critical Risk",
      textClass: "text-critical"
    };
  }
}

function buildInsight(score, inputs) {
  const parts = [];

  if (inputs.premiumFlag && score < 40) {
    parts.push("Known premium customer significantly reduces baseline risk.");
  } else if (!inputs.premiumFlag && score > 45) {
    parts.push("No premium history available, so the model leans more on behavioral and driver signals.");
  }

  if (inputs.emailAge < 6) parts.push("New email address adds risk, especially in combination with new or low-experience drivers.");
  else if (inputs.emailAge > 36) parts.push("Mature email age stabilizes identity confidence.");

  if (inputs.tenure > 36 && inputs.chargebacks < 3) parts.push("Long customer tenure with low chargebacks indicates healthy behavior.");

  if (inputs.driverAge < 23 || inputs.licenseYears < 3) parts.push("Driver age and license history suggest a relatively new driving profile.");
  else if (inputs.licenseYears > 10 && inputs.previousRentals > 5) parts.push("Experienced driver with multiple prior rentals further lowers risk.");

  if (inputs.previousRentals === 0) parts.push("First-time renter behavior is treated more cautiously by the model.");

  if (inputs.amount > 1000 || inputs.velocity > 10) parts.push("Amount and short-term velocity are above normal ranges and contribute to the score.");

  if (inputs.ipRisk > 60 || inputs.proxyFlag) parts.push("IP reputation and proxy/VPN signals are strong contributors to elevated risk.");

  if (inputs.geoDistance > 500) parts.push("Large geo distance between billing and device location increases suspicion.");

  if (inputs.phoneType === 2) parts.push("VoIP or app-based phone lines are more frequently associated with temporary or synthetic identities.");

  if (!parts.length) {
    parts.push("Signals across driver, identity, transaction, and device are balanced. The model keeps this transaction in a safe band while watching for future behavior.");
  }

  return parts.join(" ");
}

function buildPatterns(score, inputs) {
  const tags = [];

  if (inputs.velocity > 8 && inputs.amount > 400) tags.push("High-velocity + high-value pattern");
  if (inputs.emailAge < 6) tags.push("Fresh identity footprint");
  if (inputs.tenure > 36 && inputs.chargebacks <= 1) tags.push("Loyal customer behavior");
  if (inputs.driverAge < 23 || inputs.licenseYears < 3) tags.push("New / low-experience driver");
  if (inputs.licenseYears > 10 && inputs.previousRentals > 5) tags.push("Experienced driver with rental history");
  if (inputs.previousRentals === 0) tags.push("First-time renter");
  if (inputs.geoDistance > 300) tags.push("Geo anomaly: billing vs device");
  if (inputs.proxyFlag) tags.push("Proxy/VPN network");
  if (inputs.binRisk > 60) tags.push("BIN in risky cluster");
  if (inputs.phoneType === 2) tags.push("VoIP / app-based line");

  if (!tags.length && score < 25) tags.push("Stable identity & usage pattern");
  else if (!tags.length && score >= 25) tags.push("Mixed pattern · needs context");

  return tags;
}

function buildBreakdown(score, inputs) {
  const rows = [];

  if (inputs.emailAge < 6) rows.push({ signal: "Email age", status: "Very new identity", impact: "+12", className: "text-negative" });
  else if (inputs.emailAge < 18) rows.push({ signal: "Email age", status: "Moderately new", impact: "+6", className: "" });
  else rows.push({ signal: "Email age", status: "Mature history", impact: "-6", className: "text-positive" });

  if (inputs.tenure > 36 && inputs.chargebacks <= 1) rows.push({ signal: "Customer tenure", status: "Long relationship with low disputes", impact: "-10", className: "text-positive" });
  else if (inputs.tenure < 6) rows.push({ signal: "Customer tenure", status: "Brand new customer", impact: "+8", className: "text-negative" });
  else rows.push({ signal: "Customer tenure", status: "Neutral", impact: "+0", className: "" });

  if (inputs.chargebacks >= 5) rows.push({ signal: "Chargebacks", status: "Elevated disputes", impact: "+14", className: "text-critical" });
  else if (inputs.chargebacks >= 2) rows.push({ signal: "Chargebacks", status: "Some chargeback history", impact: "+6", className: "text-negative" });
  else rows.push({ signal: "Chargebacks", status: "Clean or near-clean record", impact: "-4", className: "text-positive" });

  if (inputs.driverAge < 23 || inputs.licenseYears < 3) rows.push({ signal: "Driver profile", status: "New / low-experience driver", impact: "+8", className: "text-negative" });
  else if (inputs.licenseYears > 10 && inputs.previousRentals > 5) rows.push({ signal: "Driver profile", status: "Experienced driver with rentals", impact: "-10", className: "text-positive" });
  else rows.push({ signal: "Driver profile", status: "Neutral driver profile", impact: "+0", className: "" });

  if (inputs.previousRentals === 0) rows.push({ signal: "Rental history", status: "First-time renter", impact: "+6", className: "text-negative" });
  else if (inputs.previousRentals > 5) rows.push({ signal: "Rental history", status: "Multiple prior rentals", impact: "-6", className: "text-positive" });

  if (inputs.phoneType === 2) rows.push({ signal: "Phone type", status: "VoIP / app-based line", impact: "+8", className: "text-critical" });
  else if (inputs.phoneType === 0) rows.push({ signal: "Phone type", status: "Fixed / office line", impact: "-4", className: "text-positive" });
  else rows.push({ signal: "Phone type", status: "Verified mobile", impact: "+0", className: "" });

  if (inputs.amount > 2000) rows.push({ signal: "Amount", status: "Significantly above norm", impact: "+12", className: "text-critical" });
  else if (inputs.amount > 500) rows.push({ signal: "Amount", status: "Higher than typical basket", impact: "+6", className: "text-negative" });
  else rows.push({ signal: "Amount", status: "Within normal range", impact: "+1", className: "" });

  if (inputs.velocity > 15) rows.push({ signal: "Velocity (24h)", status: "Burst of transactions", impact: "+10", className: "text-critical" });
  else if (inputs.velocity > 5) rows.push({ signal: "Velocity (24h)", status: "Higher than average", impact: "+5", className: "text-negative" });
  else rows.push({ signal: "Velocity (24h)", status: "Normal usage", impact: "+1", className: "" });

  if (inputs.deviceTrust >= 70) rows.push({ signal: "Device fingerprint", status: "Seen and trusted", impact: "-8", className: "text-positive" });
  else if (inputs.deviceTrust <= 30) rows.push({ signal: "Device fingerprint", status: "Unknown or low trust", impact: "+8", className: "text-negative" });
  else rows.push({ signal: "Device fingerprint", status: "Neutral device history", impact: "+0", className: "" });

  if (inputs.ipRisk > 65 || inputs.proxyFlag) rows.push({ signal: "IP & network", status: "Risky IP / proxy use", impact: "+12", className: "text-critical" });
  else if (inputs.ipRisk > 30) rows.push({ signal: "IP & network", status: "Moderate risk IP range", impact: "+5", className: "text-negative" });
  else rows.push({ signal: "IP & network", status: "Clean reputation", impact: "-3", className: "text-positive" });

  if (inputs.geoDistance > 800) rows.push({ signal: "Geo distance", status: "Strong billing vs device mismatch", impact: "+10", className: "text-critical" });
  else if (inputs.geoDistance > 150) rows.push({ signal: "Geo distance", status: "Noticeable mismatch", impact: "+5", className: "text-negative" });
  else rows.push({ signal: "Geo distance", status: "Aligned or nearby", impact: "+0", className: "" });

  if (inputs.premiumFlag) rows.push({ signal: "Premium profile", status: "VIP / premium customer", impact: "-10", className: "text-positive" });

  return rows;
}

/* =======================
   Slider Colors (unchanged)
======================= */

function colorForRisk(level) {
  switch (level) {
    case "low": return "#22c55e";
    case "medium": return "#eab308";
    case "high": return "#f97316";
    case "very-high": return "#ef4444";
    default: return "#64748b";
  }
}

function setSliderColor(id, level) {
  const el = document.getElementById(id);
  if (!el) return;
  const color = colorForRisk(level);
  el.style.background = `linear-gradient(90deg, ${color}, ${color})`;
}

function updateSliderRiskColors(inputs) {
  if (inputs.emailAge < 6) setSliderColor("emailAge", "very-high");
  else if (inputs.emailAge < 18) setSliderColor("emailAge", "high");
  else if (inputs.emailAge < 36) setSliderColor("emailAge", "medium");
  else setSliderColor("emailAge", "low");

  if (inputs.tenure < 6) setSliderColor("tenure", "high");
  else if (inputs.tenure < 24) setSliderColor("tenure", "medium");
  else setSliderColor("tenure", "low");

  if (inputs.chargebacks >= 5) setSliderColor("chargebacks", "very-high");
  else if (inputs.chargebacks >= 2) setSliderColor("chargebacks", "high");
  else setSliderColor("chargebacks", "low");

  if (inputs.driverAge < 23) setSliderColor("driverAge", "high");
  else if (inputs.driverAge > 70) setSliderColor("driverAge", "medium");
  else setSliderColor("driverAge", "low");

  if (inputs.licenseYears < 2) setSliderColor("licenseYears", "high");
  else if (inputs.licenseYears < 5) setSliderColor("licenseYears", "medium");
  else setSliderColor("licenseYears", "low");

  if (inputs.previousRentals === 0) setSliderColor("previousRentals", "high");
  else if (inputs.previousRentals < 3) setSliderColor("previousRentals", "medium");
  else setSliderColor("previousRentals", "low");

  if (inputs.amount > 2000) setSliderColor("amount", "very-high");
  else if (inputs.amount > 500) setSliderColor("amount", "high");
  else setSliderColor("amount", "low");

  if (inputs.velocity > 15) setSliderColor("velocity", "very-high");
  else if (inputs.velocity > 5) setSliderColor("velocity", "high");
  else setSliderColor("velocity", "low");

  if (inputs.binRisk > 70) setSliderColor("binRisk", "very-high");
  else if (inputs.binRisk > 40) setSliderColor("binRisk", "high");
  else if (inputs.binRisk > 15) setSliderColor("binRisk", "medium");
  else setSliderColor("binRisk", "low");

  if (inputs.deviceTrust >= 70) setSliderColor("deviceTrust", "low");
  else if (inputs.deviceTrust >= 40) setSliderColor("deviceTrust", "medium");
  else setSliderColor("deviceTrust", "high");

  if (inputs.ipRisk > 65) setSliderColor("ipRisk", "very-high");
  else if (inputs.ipRisk > 30) setSliderColor("ipRisk", "high");
  else setSliderColor("ipRisk", "low");

  if (inputs.geoDistance > 800) setSliderColor("geoDistance", "very-high");
  else if (inputs.geoDistance > 150) setSliderColor("geoDistance", "high");
  else setSliderColor("geoDistance", "low");
}

/* =======================
   API (Step 2) + Debounce (Step 3)
======================= */

async function scoreWithApi(payload) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`API ${res.status}: ${txt}`);
  }
  return res.json();
}

let scoreTimer = null;

function requestScoreDebounced(inputs, delay = 250) {
  if (scoreTimer) clearTimeout(scoreTimer);

  // quick UI hint while waiting
  const metaEl = document.getElementById("sphereMeta");

  // Show a neutral message only if the API call is taking noticeable time
  let slowMsgTimer = null;
  if (metaEl) {
    slowMsgTimer = setTimeout(() => {
      metaEl.textContent = "Updating score…";
    }, 600); // only show if request is slower than 600ms
  }


  scoreTimer = setTimeout(async () => {
    try {
      const data = await scoreWithApi(inputs);
      if (slowMsgTimer) clearTimeout(slowMsgTimer);


      // API returns fraud_probability (0..1). Convert to 0..100 like your UI expects.
      const score = Math.round(clamp((data.fraud_probability ?? 0.5) * 100, 0, 100));
      updateScoreUI(score, inputs);
    } catch (err) {
      console.error(err);

      // fallback to local scoring so UI still works if API cold-starts/errors
      const score = calculateRiskScore(inputs);
      updateScoreUI(score, inputs);

      const metaEl2 = document.getElementById("sphereMeta");
      if (metaEl2) metaEl2.textContent = "API temporarily unavailable, showing local estimate.";
    }
  }, delay);
}

/* =======================
   UI Rendering
======================= */

function updateScoreUI(score, inputs) {
  const tier = determineRiskTier(score);

  // Sphere pill text
  const scoreNumberEl = document.getElementById("scoreNumber");
  if (scoreNumberEl) scoreNumberEl.innerHTML = score + '<span class="unit">/ 100</span>';

  // premium readability colors per band
  let textColor = "#d9dde5";
  if (score <= 25) textColor = "#0b1a2e";
  else if (score <= 45) textColor = "#164e63";
  else if (score <= 65) textColor = "#d9dde5";

  if (scoreNumberEl) {
    scoreNumberEl.style.color = textColor;
    const unitEl = scoreNumberEl.querySelector(".unit");
    if (unitEl) unitEl.style.color = textColor;
  }

  // Text labels
  const labelEl = document.getElementById("scoreTierLabel");
  const metaEl = document.getElementById("sphereMeta");
  if (labelEl) labelEl.textContent = tier.label;
  if (metaEl) metaEl.textContent = tier.assessment;

  // Sphere color + glow
  const root = document.documentElement;
  root.style.setProperty("--sphere-color", tier.sphereColor);
  root.style.setProperty("--sphere-glow", tier.glow);

  const sphereEl = document.getElementById("riskSphere");
  if (sphereEl) {
    sphereEl.style.boxShadow =
      "0 0 18px " + tier.glow +
      ", 0 0 45px rgba(15,23,42,0.95), 0 0 85px rgba(17,94,163,0.7)";
  }

  // Risk bar marker
  const marker = document.getElementById("riskBarMarker");
  if (marker) marker.style.left = score + "%";

  // Badge
  const badge = document.getElementById("riskBadge");
  if (badge) {
    badge.textContent = "● Overall assessment: " + tier.label;
    badge.className = "risk-badge " + (tier.textClass || "");
  }

  // Insight paragraph
  const insightEl = document.getElementById("insightText");
  if (insightEl) insightEl.textContent = buildInsight(score, inputs);

  // Pattern tags
  const patternContainer = document.getElementById("patternTags");
  if (patternContainer) {
    patternContainer.innerHTML = "";
    buildPatterns(score, inputs).forEach((p) => {
      const span = document.createElement("span");
      span.className = "pattern-tag";
      span.textContent = p;
      patternContainer.appendChild(span);
    });
  }

  // Breakdown table
  const breakdownBody = document.getElementById("breakdownBody");
  if (breakdownBody) {
    breakdownBody.innerHTML = "";
    buildBreakdown(score, inputs).forEach((row) => {
      const tr = document.createElement("tr");
      const tdSignal = document.createElement("td");
      const tdStatus = document.createElement("td");
      const tdImpact = document.createElement("td");

      tdSignal.textContent = row.signal;
      tdStatus.textContent = row.status;
      tdImpact.textContent = row.impact;
      tdImpact.style.textAlign = "right";
      if (row.className) tdImpact.className = row.className;

      tr.appendChild(tdSignal);
      tr.appendChild(tdStatus);
      tr.appendChild(tdImpact);
      breakdownBody.appendChild(tr);
    });
  }

  // Per-field slider risk colors
  updateSliderRiskColors(inputs);
}

/* =======================
   Main UI Update
======================= */

function readInputsFromDom() {
  return {
    emailAge: Number(document.getElementById("emailAge").value),
    tenure: Number(document.getElementById("tenure").value),
    chargebacks: Number(document.getElementById("chargebacks").value),
    premiumFlag: Number(document.getElementById("premiumFlag").value),
    driverAge: Number(document.getElementById("driverAge").value),
    licenseYears: Number(document.getElementById("licenseYears").value),
    previousRentals: Number(document.getElementById("previousRentals").value),
    phoneType: Number(document.getElementById("phoneType").value),
    amount: Number(document.getElementById("amount").value),
    velocity: Number(document.getElementById("velocity").value),
    binRisk: Number(document.getElementById("binRisk").value),
    oddHour: Number(document.getElementById("oddHour").value),
    deviceTrust: Number(document.getElementById("deviceTrust").value),
    ipRisk: Number(document.getElementById("ipRisk").value),
    geoDistance: Number(document.getElementById("geoDistance").value),
    proxyFlag: Number(document.getElementById("proxyFlag").value)
  };
}

function updateLabels(inputs) {
  document.getElementById("emailAgeDisplay").textContent = inputs.emailAge + " months";
  document.getElementById("tenureDisplay").textContent = inputs.tenure + " months";
  document.getElementById("chargebacksDisplay").textContent = inputs.chargebacks + " %";
  document.getElementById("driverAgeDisplay").textContent = inputs.driverAge + " years";
  document.getElementById("licenseYearsDisplay").textContent = inputs.licenseYears + " years";
  document.getElementById("previousRentalsDisplay").textContent = inputs.previousRentals + " rentals";
  document.getElementById("amountDisplay").textContent = "$" + inputs.amount;
  document.getElementById("velocityDisplay").textContent = inputs.velocity + " tx";
  document.getElementById("binRiskDisplay").textContent = inputs.binRisk + " / 100";
  document.getElementById("deviceTrustDisplay").textContent = inputs.deviceTrust + " / 100";
  document.getElementById("ipRiskDisplay").textContent = inputs.ipRisk + " / 100";
  document.getElementById("geoDistanceDisplay").textContent = inputs.geoDistance + " km";
}

function updateUI() {
  const inputs = readInputsFromDom();
  updateLabels(inputs);

  // Step 3: call API (debounced), with local fallback inside
  requestScoreDebounced(inputs, 500);
}

/* =======================
   INIT
======================= */

window.addEventListener("DOMContentLoaded", () => {
  const inputsAll = document.querySelectorAll(
    "#emailAge, #tenure, #chargebacks, #premiumFlag, #driverAge, #licenseYears, #previousRentals, #phoneType, #amount, #velocity, #binRisk, #oddHour, #deviceTrust, #ipRisk, #geoDistance, #proxyFlag"
  );

  inputsAll.forEach((input) => {
    input.addEventListener("input", updateUI);
    input.addEventListener("change", updateUI);
  });

  updateUI();
});
