/**
 * student-module/utils/chartHelpers.js
 * -------------------------------------
 * Helpers for formatting data into chart-ready structures.
 * Charts are rendered using react-native-svg (already installed).
 */

/**
 * Build monthly attendance trend data from timeline entries.
 * @param {Array} timelineData - Array of {date, status, subject, ...}
 * @param {number} month
 * @param {number} year
 * @returns {Array<{day: number, present: number, absent: number, total: number}>}
 */
export const buildDailyTrend = (timelineData, month, year) => {
  const daysInMonth = new Date(year, month, 0).getDate();
  const dailyMap = {};

  for (let d = 1; d <= daysInMonth; d++) {
    dailyMap[d] = { day: d, present: 0, absent: 0, total: 0 };
  }

  timelineData.forEach(entry => {
    const date = new Date(entry.date);
    const day = date.getDate();
    if (dailyMap[day]) {
      dailyMap[day].total += 1;
      if (entry.status === 'Present') {
        dailyMap[day].present += 1;
      } else {
        dailyMap[day].absent += 1;
      }
    }
  });

  return Object.values(dailyMap).filter(d => d.total > 0);
};

/**
 * Build weekly attendance data from timeline entries.
 * @param {Array} timelineData
 * @returns {Array<{week: string, present: number, absent: number, total: number, percentage: number}>}
 */
export const buildWeeklyTrend = (timelineData) => {
  const weekMap = {};
  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  timelineData.forEach(entry => {
    const date = new Date(entry.date);
    const dayName = dayNames[date.getDay()];
    if (!weekMap[dayName]) {
      weekMap[dayName] = { week: dayName, present: 0, absent: 0, total: 0 };
    }
    weekMap[dayName].total += 1;
    if (entry.status === 'Present') {
      weekMap[dayName].present += 1;
    } else {
      weekMap[dayName].absent += 1;
    }
  });

  // Return in Mon-Sun order
  const order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  return order
    .filter(day => weekMap[day])
    .map(day => ({
      ...weekMap[day],
      percentage: weekMap[day].total > 0
        ? Math.round((weekMap[day].present / weekMap[day].total) * 100)
        : 0,
    }));
};

/**
 * Build subject comparison data from report subjects.
 * @param {Array} subjects - From /student/report -> subjects[]
 * @returns {Array<{name: string, percentage: number, present: number, total: number, color: string}>}
 */
export const buildSubjectComparison = (subjects) => {
  const colors = [
    '#2563eb', '#7c3aed', '#059669', '#ea580c', '#dc2626',
    '#0891b2', '#4f46e5', '#ca8a04', '#be185d', '#16a34a',
  ];

  return (subjects || []).map((subject, index) => ({
    name: subject.name || 'Unknown',
    shortName: (subject.name || 'Unknown').length > 12
      ? (subject.name || '').substring(0, 10) + '…'
      : subject.name || 'Unknown',
    percentage: subject.percentage || 0,
    present: subject.present || 0,
    total: subject.total_classes || 0,
    absent: subject.absent || 0,
    color: colors[index % colors.length],
  }));
};

/**
 * Calculate overall present vs absent totals.
 * @param {object} report - From /student/report
 * @returns {{present: number, absent: number, total: number}}
 */
export const buildPresentAbsent = (report) => {
  const present = report?.total_present || 0;
  const total = report?.total_classes || 0;
  const absent = total - present;
  return { present, absent: Math.max(absent, 0), total };
};

/**
 * Build monthly trend from analytics data.
 * @param {Array} monthlyData - [{month, year, percentage, present, total}]
 * @returns {Array}
 */
export const buildMonthlyTrend = (monthlyData) => {
  const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return (monthlyData || []).map(item => ({
    ...item,
    label: monthNames[(item.month || 1) - 1] || '',
  }));
};
