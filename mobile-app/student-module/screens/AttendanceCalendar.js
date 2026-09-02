/**
 * student-module/screens/AttendanceCalendar.js
 * -----------------------------------------------
 * Feature 3: Monthly attendance calendar with day-detail view.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  RefreshControl, Modal,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import { ListSkeleton } from '../components/LoadingSkeleton';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import studentApi from '../services/studentApi';
import { getMonthName, getDaysInMonth, getFirstDayOfMonth, getDayName } from '../utils/dateUtils';

const DAY_HEADERS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

const AttendanceCalendar = ({ navigateTo }) => {
  const now = new Date();
  const [month, setMonth] = useState(now.getMonth() + 1);
  const [year, setYear] = useState(now.getFullYear());
  const [timeline, setTimeline] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [selectedDay, setSelectedDay] = useState(null);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const res = await studentApi.getTimeline(month, year);
      if (res.status === 'success') {
        setTimeline(res.data || []);
      }
    } catch (err) {
      setError(err.message || 'Failed to load timeline');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [month, year]);

  useEffect(() => { setLoading(true); loadData(); }, [loadData]);

  const onRefresh = useCallback(() => { setRefreshing(true); loadData(); }, [loadData]);

  const prevMonth = () => {
    if (month === 1) { setMonth(12); setYear(y => y - 1); }
    else setMonth(m => m - 1);
    setSelectedDay(null);
  };

  const nextMonth = () => {
    if (month === 12) { setMonth(1); setYear(y => y + 1); }
    else setMonth(m => m + 1);
    setSelectedDay(null);
  };

  // Build calendar grid data
  const daysInMonth = getDaysInMonth(month, year);
  const firstDay = getFirstDayOfMonth(month, year);

  // Group timeline entries by day
  const dayMap = {};
  timeline.forEach(entry => {
    const date = new Date(entry.date);
    const day = date.getDate();
    if (!dayMap[day]) dayMap[day] = [];
    dayMap[day].push(entry);
  });

  // Get status color for a day
  const getDayStatus = (day) => {
    const entries = dayMap[day];
    if (!entries || entries.length === 0) return null;
    const hasPresent = entries.some(e => e.status === 'Present');
    const hasAbsent = entries.some(e => e.status === 'Absent');
    const hasManual = entries.some(e => e.marked_by === 'Manual');
    if (hasManual) return 'manual';
    if (hasPresent && !hasAbsent) return 'present';
    if (hasAbsent && !hasPresent) return 'absent';
    return 'mixed';
  };

  const statusColors = {
    present: Colors.success,
    absent: Colors.danger,
    manual: Colors.warning,
    mixed: Colors.orange,
  };

  const statusBg = {
    present: Colors.successLight,
    absent: Colors.dangerLight,
    manual: Colors.warningLight,
    mixed: Colors.orangeLight,
  };

  // Build grid cells
  const gridCells = [];
  // Empty cells before first day
  for (let i = 0; i < firstDay; i++) {
    gridCells.push({ type: 'empty', key: `empty-${i}` });
  }
  // Day cells
  for (let d = 1; d <= daysInMonth; d++) {
    const status = getDayStatus(d);
    gridCells.push({ type: 'day', day: d, status, key: `day-${d}` });
  }

  const selectedEntries = selectedDay ? (dayMap[selectedDay] || []) : [];

  if (loading && !refreshing) return <ListSkeleton count={3} />;

  return (
    <ScrollView
      style={calStyles.container}
      contentContainerStyle={calStyles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={[Colors.primary]} />}
      showsVerticalScrollIndicator={false}
    >
      {error ? (
        <ErrorState message={error} onRetry={loadData} />
      ) : (
        <>
          {/* Month Navigation */}
          <View style={calStyles.monthNav}>
            <TouchableOpacity onPress={prevMonth} style={calStyles.navBtn}>
              <Ionicons name="chevron-back" size={20} color={Colors.primary} />
            </TouchableOpacity>
            <Text style={calStyles.monthTitle}>{getMonthName(month)} {year}</Text>
            <TouchableOpacity onPress={nextMonth} style={calStyles.navBtn}>
              <Ionicons name="chevron-forward" size={20} color={Colors.primary} />
            </TouchableOpacity>
          </View>

          {/* Legend */}
          <View style={calStyles.legend}>
            <View style={calStyles.legendItem}>
              <View style={[calStyles.legendDot, { backgroundColor: Colors.success }]} />
              <Text style={calStyles.legendText}>Present</Text>
            </View>
            <View style={calStyles.legendItem}>
              <View style={[calStyles.legendDot, { backgroundColor: Colors.danger }]} />
              <Text style={calStyles.legendText}>Absent</Text>
            </View>
            <View style={calStyles.legendItem}>
              <View style={[calStyles.legendDot, { backgroundColor: Colors.warning }]} />
              <Text style={calStyles.legendText}>Manual</Text>
            </View>
            <View style={calStyles.legendItem}>
              <View style={[calStyles.legendDot, { backgroundColor: '#d1d5db' }]} />
              <Text style={calStyles.legendText}>No Data</Text>
            </View>
          </View>

          {/* Calendar Grid */}
          <View style={calStyles.calendarCard}>
            {/* Day headers */}
            <View style={calStyles.headerRow}>
              {DAY_HEADERS.map(d => (
                <Text key={d} style={calStyles.headerCell}>{d}</Text>
              ))}
            </View>

            {/* Day grid */}
            <View style={calStyles.grid}>
              {gridCells.map(cell => {
                if (cell.type === 'empty') {
                  return <View key={cell.key} style={calStyles.cell} />;
                }
                const isSelected = selectedDay === cell.day;
                const hasBg = cell.status && statusBg[cell.status];
                return (
                  <TouchableOpacity
                    key={cell.key}
                    style={[
                      calStyles.cell,
                      hasBg && { backgroundColor: hasBg },
                      isSelected && calStyles.cellSelected,
                    ]}
                    onPress={() => setSelectedDay(cell.day === selectedDay ? null : cell.day)}
                    activeOpacity={0.7}
                  >
                    <Text style={[
                      calStyles.dayText,
                      cell.status && { color: statusColors[cell.status], fontWeight: '700' },
                      isSelected && { color: Colors.textWhite },
                    ]}>
                      {cell.day}
                    </Text>
                    {cell.status && (
                      <View style={[
                        calStyles.statusDot,
                        { backgroundColor: isSelected ? Colors.textWhite : statusColors[cell.status] },
                      ]} />
                    )}
                  </TouchableOpacity>
                );
              })}
            </View>
          </View>

          {/* Selected Day Details */}
          {selectedDay !== null && (
            <View style={calStyles.detailCard}>
              <Text style={calStyles.detailTitle}>
                {selectedDay} {getMonthName(month)} {year}
              </Text>
              {selectedEntries.length === 0 ? (
                <Text style={calStyles.noData}>No attendance recorded</Text>
              ) : (
                selectedEntries.map((entry, i) => (
                  <View key={i} style={calStyles.detailRow}>
                    <View style={calStyles.detailLeft}>
                      <Text style={calStyles.detailSubject}>{entry.subject}</Text>
                      <Text style={calStyles.detailInfo}>
                        {entry.faculty_name && `${entry.faculty_name} • `}{entry.time}
                      </Text>
                    </View>
                    <View style={calStyles.detailRight}>
                      <View style={[
                        calStyles.detailBadge,
                        { backgroundColor: entry.status === 'Present' ? Colors.successLight : Colors.dangerLight },
                      ]}>
                        <Text style={{
                          color: entry.status === 'Present' ? Colors.success : Colors.danger,
                          fontSize: FontSizes.xs, fontWeight: 'bold',
                        }}>
                          {entry.status}
                        </Text>
                      </View>
                      {entry.marked_by === 'Manual' && (
                        <Text style={calStyles.manualTag}>✏ Manual</Text>
                      )}
                    </View>
                  </View>
                ))
              )}
            </View>
          )}

          {timeline.length === 0 && !selectedDay && (
            <EmptyState
              icon="📅"
              title="No Attendance Data"
              subtitle={`No attendance recorded for ${getMonthName(month)} ${year}`}
            />
          )}
        </>
      )}
      <View style={{ height: 24 }} />
    </ScrollView>
  );
};

const calStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  content: { padding: Spacing.lg },
  // Month Nav
  monthNav: {
    flexDirection: 'row', justifyContent: 'space-between',
    alignItems: 'center', marginBottom: Spacing.md,
  },
  navBtn: {
    padding: Spacing.sm, backgroundColor: Colors.primaryLight,
    borderRadius: BorderRadius.md,
  },
  monthTitle: { fontSize: FontSizes.xl, fontWeight: 'bold', color: Colors.text },
  // Legend
  legend: {
    flexDirection: 'row', justifyContent: 'space-around',
    marginBottom: Spacing.md,
    backgroundColor: Colors.surface, borderRadius: BorderRadius.md,
    padding: Spacing.sm,
  },
  legendItem: { flexDirection: 'row', alignItems: 'center' },
  legendDot: { width: 8, height: 8, borderRadius: 4, marginRight: 4 },
  legendText: { fontSize: 10, color: Colors.textTertiary },
  // Calendar
  calendarCard: {
    backgroundColor: Colors.surface, borderRadius: BorderRadius.xl,
    padding: Spacing.md, borderWidth: 1, borderColor: Colors.border,
    marginBottom: Spacing.lg,
  },
  headerRow: { flexDirection: 'row', marginBottom: Spacing.xs },
  headerCell: {
    flex: 1, textAlign: 'center', fontSize: FontSizes.xs,
    color: Colors.textMuted, fontWeight: '600', paddingVertical: 4,
  },
  grid: { flexDirection: 'row', flexWrap: 'wrap' },
  cell: {
    width: '14.28%', aspectRatio: 1,
    alignItems: 'center', justifyContent: 'center',
    borderRadius: BorderRadius.md, marginVertical: 1,
  },
  cellSelected: { backgroundColor: Colors.primary },
  dayText: { fontSize: FontSizes.md, color: Colors.text },
  statusDot: { width: 4, height: 4, borderRadius: 2, marginTop: 2 },
  // Detail
  detailCard: {
    backgroundColor: Colors.surface, borderRadius: BorderRadius.xl,
    padding: Spacing.lg, borderWidth: 1, borderColor: Colors.border,
  },
  detailTitle: { fontSize: FontSizes.lg, fontWeight: 'bold', color: Colors.text, marginBottom: Spacing.md },
  noData: { fontSize: FontSizes.sm, color: Colors.textMuted, textAlign: 'center', paddingVertical: Spacing.lg },
  detailRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingVertical: Spacing.md, borderBottomWidth: 1, borderBottomColor: Colors.borderLight,
  },
  detailLeft: { flex: 1 },
  detailSubject: { fontSize: FontSizes.md, fontWeight: '600', color: Colors.text },
  detailInfo: { fontSize: FontSizes.xs, color: Colors.textTertiary, marginTop: 2 },
  detailRight: { alignItems: 'flex-end' },
  detailBadge: { paddingHorizontal: Spacing.sm, paddingVertical: 2, borderRadius: BorderRadius.sm },
  manualTag: { fontSize: 9, color: Colors.warning, marginTop: 2 },
});

export default AttendanceCalendar;
