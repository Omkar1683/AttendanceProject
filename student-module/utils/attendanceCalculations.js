/**
 * student-module/utils/attendanceCalculations.js
 * -----------------------------------------------
 * Pure math helpers for attendance prediction and calculations.
 */

/**
 * Calculate attendance percentage.
 */
export const calculatePercentage = (present, total) => {
  if (total === 0) return 0;
  return Math.round((present / total) * 100 * 100) / 100;
};

/**
 * Calculate what the percentage would be if the student misses the next N classes.
 * @param {number} present - Current present count
 * @param {number} total - Current total count
 * @param {number} missCount - Number of classes to miss
 * @returns {number} Projected percentage
 */
export const percentageAfterMissing = (present, total, missCount = 1) => {
  const newTotal = total + missCount;
  if (newTotal === 0) return 0;
  return Math.round((present / newTotal) * 100 * 100) / 100;
};

/**
 * Calculate the minimum number of consecutive classes the student must attend
 * to reach a target percentage.
 *
 * Formula:
 *   We need (present + x) / (total + x) >= target/100
 *   Solving: x >= (target * total - 100 * present) / (100 - target)
 *
 * @param {number} present - Current present count
 * @param {number} total - Current total count
 * @param {number} targetPercent - Target percentage (e.g. 75)
 * @returns {number|null} Number of classes needed, 0 if already achieved, null if impossible (target >= 100)
 */
export const classesNeededForTarget = (present, total, targetPercent) => {
  if (targetPercent >= 100) return null; // Impossible to reach 100% if any absences
  
  const currentPct = calculatePercentage(present, total);
  if (currentPct >= targetPercent) return 0; // Already achieved
  
  // (present + x) / (total + x) >= targetPercent / 100
  // present + x >= (targetPercent / 100) * (total + x)
  // present + x >= targetPercent * total / 100 + targetPercent * x / 100
  // x - (targetPercent * x / 100) >= (targetPercent * total / 100) - present
  // x * (1 - targetPercent/100) >= (targetPercent * total / 100) - present
  // x >= ((targetPercent * total / 100) - present) / (1 - targetPercent/100)
  
  const numerator = (targetPercent * total / 100) - present;
  const denominator = 1 - (targetPercent / 100);
  
  if (denominator <= 0) return null;
  
  const needed = Math.ceil(numerator / denominator);
  return Math.max(needed, 0);
};

/**
 * Generate prediction data for standard targets.
 * @param {number} present
 * @param {number} total
 * @returns {Array<{target: number, needed: number|null, achieved: boolean}>}
 */
export const generatePredictions = (present, total) => {
  const targets = [75, 80, 85, 90];
  return targets.map(target => {
    const needed = classesNeededForTarget(present, total, target);
    return {
      target,
      needed,
      achieved: needed === 0,
      impossible: needed === null,
    };
  });
};

/**
 * Get a human-readable attendance status.
 */
export const getAttendanceStatus = (percentage) => {
  if (percentage >= 90) return { label: 'Excellent', color: 'success' };
  if (percentage >= 75) return { label: 'Safe', color: 'success' };
  if (percentage >= 60) return { label: 'Warning', color: 'warning' };
  return { label: 'Defaulter', color: 'danger' };
};
