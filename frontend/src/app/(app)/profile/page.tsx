"use client"

import { ActivityTabs } from "@/components/profile/activity-tabs"
import { PersonalRecords } from "@/components/profile/personal-records"
import { ProfileHero } from "@/components/profile/profile-hero"
import { RecentActivity } from "@/components/profile/recent-activity"
import { SettingsSheet } from "@/components/profile/settings-sheet"
import { StatsSummary } from "@/components/profile/stats-summary"

export default function ProfilePage() {
  return (
    <div className="mx-auto max-w-lg space-y-5 px-4 py-4">
      <ProfileHero />
      <StatsSummary />
      <ActivityTabs activityContent={<RecentActivity />} recordsContent={<PersonalRecords />} />
      <SettingsSheet />
    </div>
  )
}
