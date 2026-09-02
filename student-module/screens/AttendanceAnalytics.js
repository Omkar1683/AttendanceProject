/**
 * student-module/screens/AttendanceAnalytics.js
 * ------------------------------------------------
 * Feature 6: Charts and analytics using react-native-svg.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, ScrollView, StyleSheet, RefreshControl, Dimensions,
} from 'react-native';
import Svg, { Rect, Circle, Line, Text as SvgText, G, Path } from 'react-native-svg';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import { DashboardSkeleton } from '../components/LoadingSkeleton';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import studentApi from '../services/studentApi';
import { buildSubjectComparison, buildMonthlyTrend } from '../utils/chartHelpers';

const SCREEN_WIDTH = Dimensions.get('window').width;
const CHART_WIDTH = SCREEN_WIDTH - 64; // padding
const CHART_HEIGHT = 180;
const BAR_WIDTH = 24;

const AttendanceAnalytics = ({ navigateTo }) => {
  const [analytics, setAnalytics] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [analyticsRes, reportRes] = await Promise.all([
        studentApi.getAnalytics(),
        studentApi.getReport(),
      ]);
      if (analyticsRes.status === 'success') setAnalytics(analyticsRes.data);
      if (reportRes.status === 'success') setReport(reportRes.data);
    } catch (err) {
      setError(err.message || 'Failed to load analytics');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const onRefresh = useCallback(() => { setRefreshing(true); loadData(); }, [loadData]);

  if (loading) return <DashboardSkeleton />;
  if (error) return <ErrorState message={error} onRetry={loadData} />;

  const monthlyTrend = buildMonthlyTrend(analytics?.monthly_trend || []);
  const weeklyTrend = analytics?.weekly_trend || [];
  const presentAbsent = analytics?.present_absent || { present: 0, absent: 0, total: 0 };
  const subjects = buildSubjectComparison(report?.subjects || []);

  const hasData = monthlyTrend.some(m => m.total > 0) || subjects.length > 0;

  if (!hasData) {
    return (
      <ScrollView
        style={aStyles.container}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      >
        <EmptyState icon="📊" title="No Analytics Data" subtitle="Attend some classes to see your analytics here." />
      </ScrollView>
    );
  }

  // ── Pie Chart Data ──────────────────────────────────────────────────────────
  const pieTotal = presentAbsent.total || 1;
  const presentAngle = (presentAbsent.present / pieTotal) * 360;

  const describeArc = (cx, cy, r, startAngle, endAngle) => {
    const start = polarToCartesian(cx, cy, r, endAngle);
    const end = polarToCartesian(cx, cy, r, startAngle);
    const largeArcFlag = endAngle - startAngle > 180 ? 1 : 0;
    return `M ${cx} ${cy} L ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y} Z`;
  };

  const polarToCartesian = (cx, cy, r, angle) => {
    const rad = ((angle - 90) * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  };

  return (
    <ScrollView
      style={aStyles.container}
      contentContainerStyle={aStyles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={[Colors.primary]} />}
      showsVerticalScrollIndicator={false}
    >
      {/* ── Attendance Donut ────────────────────────────────────────────── */}
      <View style={aStyles.card}>
        <Text style={aStyles.cardLabel}>ATTENDANCE OVERVIEW</Text>
        <View style={aStyles.donutRow}>
          <Svg width={120} height={120} viewBox="0 0 120 120">
            <Circle cx={60} cy={60} r={50} fill={Colors.dangerLight} />
            {presentAngle > 0 && (
              <Path
                d={describeArc(60, 60, 50, 0, Math.min(presentAngle, 359.99))}
                fill={Colors.success}
              />
            )}
            <Circle cx={60} cy={60} r={30} fill={Colors.surface} />
            <SvgText
              x={60} y={57} textAnchor="middle" fontSize={18}
              fontWeight="bold" fill={Colors.text}
            >
              {report?.overall_percentage || 0}%
            </SvgText>
            <SvgText
              x={60} y={72} textAnchor="middle" fontSize={10}
              fill={Colors.textMuted}
            >
              Overall
            </SvgText>
          </Svg>
          <View style={aStyles.donutLegend}>
            <View style={aStyles.legendRow}>
              <View style={[aStyles.legendCircle, { backgroundColor: Colors.success }]} />
              <Text style={aStyles.legendLabel}>Present</Text>
              <Text style={aStyles.legendValue}>{presentAbsent.present}</Text>
            </View>
            <View style={aStyles.legendRow}>
              <View style={[aStyles.legendCircle, { backgroundColor: Colors.danger }]} />
              <Text style={aStyles.legendLabel}>Absent</Text>
              <Text style={aStyles.legendValue}>{presentAbsent.absent}</Text>
            </View>
            <View style={[aStyles.legendRow, { borderTopWidth: 1, borderTopColor: Colors.borderLight, paddingTop: 8 }]}>
              <Text style={[aStyles.legendLabel, { fontWeight: '600' }]}>Total</Text>
              <Text style={[aStyles.legendValue, { fontWeight: '600' }]}>{presentAbsent.total}</Text>
            </View>
          </View>
        </View>
      </View>

      {/* ── Monthly Trend Bar Chart ────────────────────────────────────── */}
      {monthlyTrend.length > 0 && (
        <View style={aStyles.card}>
          <Text style={aStyles.cardLabel}>MONTHLY TREND</Text>
          <Svg width={CHART_WIDTH} height={CHART_HEIGHT + 30} viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT + 30}`}>
            {/* Grid lines */}
            {[0, 25, 50, 75, 100].map(pct => {
              const y = CHART_HEIGHT - (pct / 100) * CHART_HEIGHT;
              return (
                <G key={pct}>
                  <Line x1={30} y1={y} x2={CHART_WIDTH} y2={y} stroke={Colors.borderLight} strokeWidth={1} />
                  <SvgText x={0} y={y + 4} fontSize={9} fill={Colors.textMuted}>{pct}%</SvgText>
                </G>
              );
            })}
            {/* 75% threshold line */}
            <Line
              x1={30} y1={CHART_HEIGHT - 0.75 * CHART_HEIGHT}
              x2={CHART_WIDTH} y2={CHART_HEIGHT - 0.75 * CHART_HEIGHT}
              stroke={Colors.danger} strokeWidth={1} strokeDasharray="4,4"
            />
            {/* Bars */}
            {monthlyTrend.map((item, i) => {
              const barH = (item.percentage / 100) * CHART_HEIGHT;
              const x = 40 + i * ((CHART_WIDTH - 50) / monthlyTrend.length);
              const y = CHART_HEIGHT - barH;
              const barColor = item.percentage >= 75 ? Colors.success : Colors.danger;
              return (
                <G key={i}>
                  <Rect x={x} y={y} width={BAR_WIDTH} height={barH} rx={4} fill={barColor} opacity={0.85} />
                  <SvgText x={x + BAR_WIDTH / 2} y={CHART_HEIGHT + 14} textAnchor="middle" fontSize={9} fill={Colors.textMuted}>
                    {item.label}
                  </SvgText>
                  <SvgText x={x + BAR_WIDTH / 2} y={y - 4} textAnchor="middle" fontSize={8} fill={barColor} fontWeight="bold">
                    {item.percentage}%
                  </SvgText>
                </G>
              );
            })}
          </Svg>
        </View>
      )}

      {/* ── Weekly Trend ───────────────────────────────────────────────── */}
      {weeklyTrend.length > 0 && (
        <View style={aStyles.card}>
          <Text style={aStyles.cardLabel}>WEEKLY PATTERN</Text>
          {weeklyTrend.map((item, i) => (
            <View key={i} style={aStyles.weekRow}>
              <Text style={aStyles.weekDay}>{item.day}</Text>
              <View style={aStyles.weekBarBg}>
                <View style={[
                  aStyles.weekBarFill,
                  {
                    width: `${item.percentage}%`,
                    backgroundColor: item.percentage >= 75 ? Colors.success : Colors.danger,
                  },
                ]} />
              </View>
              <Text style={[aStyles.weekPct, {
                color: item.percentage >= 75 ? Colors.success : Colors.danger,
              }]}>
                {item.percentage}%
              </Text>
            </View>
          ))}
        </View>
      )}

      {/* ── Subject Comparison ─────────────────────────────────────────── */}
      {subjects.length > 0 && (
        <View style={aStyles.card}>
          <Text style={aStyles.cardLabel}>SUBJECT COMPARISON</Text>
          <Svg width={CHART_WIDTH} height={subjects.length * 36 + 10} viewBox={`0 0 ${CHART_WIDTH} ${subjects.length * 36 + 10}`}>
            {subjects.map((subj, i) => {
              const y = i * 36 + 5;
              const barW = Math.max((subj.percentage / 100) * (CHART_WIDTH - 100), 2);
              return (
                <G key={i}>
                  <SvgText x={0} y={y + 12} fontSize={10} fill={Colors.text} fontWeight="500">
                    {subj.shortName}
                  </SvgText>
                  <Rect x={0} y={y + 18} width={CHART_WIDTH - 50} height={8} rx={4} fill={Colors.surfaceAlt} />
                  <Rect x={0} y={y + 18} width={barW} height={8} rx={4} fill={subj.color} />
                  <SvgText x={CHART_WIDTH - 10} y={y + 26} textAnchor="end" fontSize={10} fill={subj.color} fontWeight="bold">
                    {subj.percentage}%
                  </SvgText>
                </G>
              );
            })}
          </Svg>
        </View>
      )}

      <View style={{ height: 24 }} />
    </ScrollView>
  );
};

const aStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  content: { padding: Spacing.lg },
  card: {
    backgroundColor: Colors.surface, borderRadius: BorderRadius.xl,
    padding: Spacing.lg, marginBottom: Spacing.lg,
    borderWidth: 1, borderColor: Colors.border,
  },
  cardLabel: {
    fontSize: FontSizes.xs, fontWeight: 'bold', color: Colors.textMuted,
    marginBottom: Spacing.md, letterSpacing: 1,
  },
  // Donut
  donutRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.xl },
  donutLegend: { flex: 1 },
  legendRow: {
    flexDirection: 'row', alignItems: 'center', marginBottom: Spacing.sm,
  },
  legendCircle: { width: 10, height: 10, borderRadius: 5, marginRight: Spacing.sm },
  legendLabel: { flex: 1, fontSize: FontSizes.sm, color: Colors.textSecondary },
  legendValue: { fontSize: FontSizes.sm, fontWeight: '600', color: Colors.text },
  // Weekly
  weekRow: {
    flexDirection: 'row', alignItems: 'center', marginBottom: Spacing.sm,
  },
  weekDay: { width: 36, fontSize: FontSizes.sm, color: Colors.textSecondary, fontWeight: '500' },
  weekBarBg: {
    flex: 1, height: 8, backgroundColor: Colors.surfaceAlt,
    borderRadius: 4, overflow: 'hidden', marginHorizontal: Spacing.sm,
  },
  weekBarFill: { height: '100%', borderRadius: 4 },
  weekPct: { width: 40, textAlign: 'right', fontSize: FontSizes.sm, fontWeight: '600' },
});

export default AttendanceAnalytics;
