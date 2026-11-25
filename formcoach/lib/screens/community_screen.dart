import 'package:flutter/material.dart';
import '../data/social_mock.dart';
import '../models/social.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'group_detail_screen.dart';

class CommunityScreen extends StatelessWidget {
  const CommunityScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Column(
        children: [
          const Padding(
            padding: EdgeInsets.fromLTRB(16, 8, 16, 0),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text('Community',
                  style: TextStyle(fontSize: 28, fontWeight: FontWeight.w800)),
            ),
          ),
          const TabBar(
            indicatorColor: AppColors.accent,
            labelColor: AppColors.text,
            unselectedLabelColor: AppColors.textDim,
            dividerColor: AppColors.line,
            tabs: [Tab(text: 'Feed'), Tab(text: 'Friends'), Tab(text: 'Groups')],
          ),
          const Expanded(
            child: TabBarView(children: [_FeedTab(), _FriendsTab(), _GroupsTab()]),
          ),
        ],
      ),
    );
  }
}

class Avatar extends StatelessWidget {
  final Person person;
  final double size;
  const Avatar(this.person, {super.key, this.size = 44});

  @override
  Widget build(BuildContext context) => CircleAvatar(
        radius: size / 2,
        backgroundColor: AppColors.surfaceHi,
        foregroundColor: AppColors.text,
        child: Text(person.initials,
            style: TextStyle(
                fontWeight: FontWeight.w700, fontSize: size * 0.34)),
      );
}

class RoleBadge extends StatelessWidget {
  final GroupRole role;
  const RoleBadge(this.role, {super.key});

  @override
  Widget build(BuildContext context) {
    if (role == GroupRole.member) return const SizedBox.shrink();
    final c = role == GroupRole.owner ? AppColors.accent : AppColors.warn;
    return Container(
      margin: const EdgeInsets.only(left: 8),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: c.withValues(alpha: 0.16),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(role.label.toUpperCase(),
          style: TextStyle(
              color: c, fontSize: 10, fontWeight: FontWeight.w800, letterSpacing: 0.8)),
    );
  }
}

// ---------------- Feed ----------------
class _FeedTab extends StatelessWidget {
  const _FeedTab();

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(AppSpace.md),
      children: [
        Panel(
          child: Row(
            children: [
              const Avatar(me),
              const SizedBox(width: 12),
              const Expanded(
                child: Text('Share a set or ask for a form check',
                    style: TextStyle(color: AppColors.textDim)),
              ),
              const Icon(Icons.videocam_outlined, color: AppColors.accent),
            ],
          ),
        ),
        const SizedBox(height: 12),
        for (final p in feed)
          Padding(padding: const EdgeInsets.only(bottom: 12), child: _PostCard(p)),
      ],
    );
  }
}

class _PostCard extends StatefulWidget {
  final Post post;
  const _PostCard(this.post);

  @override
  State<_PostCard> createState() => _PostCardState();
}

class _PostCardState extends State<_PostCard> {
  bool liked = false;

  @override
  Widget build(BuildContext context) {
    final p = widget.post;
    return Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Avatar(p.author),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(p.author.name,
                      style: const TextStyle(fontWeight: FontWeight.w700)),
                  Text(
                      p.groupName == null ? p.when : '${p.when}  -  ${p.groupName}',
                      style: const TextStyle(
                          color: AppColors.textDim, fontSize: 12)),
                ],
              ),
            ),
          ]),
          const SizedBox(height: 12),
          Text(p.text),
          if (p.exercise != null) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppColors.surfaceHi,
                borderRadius: BorderRadius.circular(AppRadius.md),
              ),
              child: Row(children: [
                ScoreRing(score: p.formScore!, size: 48, stroke: 5, showLabel: false),
                const SizedBox(width: 12),
                Expanded(
                  child: Text('${p.exercise}  -  ${p.reps} reps',
                      style: const TextStyle(fontWeight: FontWeight.w600)),
                ),
                Text('${p.formScore!.round()}',
                    style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                        color: scoreColor(p.formScore!))),
              ]),
            ),
          ],
          const SizedBox(height: 8),
          Row(children: [
            IconButton(
              onPressed: () => setState(() => liked = !liked),
              icon: Icon(liked ? Icons.favorite : Icons.favorite_border,
                  color: liked ? AppColors.bad : AppColors.textDim),
            ),
            Text('${p.likes + (liked ? 1 : 0)}',
                style: const TextStyle(color: AppColors.textDim)),
            const SizedBox(width: 12),
            const Icon(Icons.chat_bubble_outline,
                size: 20, color: AppColors.textDim),
          ]),
        ],
      ),
    );
  }
}

// ---------------- Friends ----------------
class _FriendsTab extends StatelessWidget {
  const _FriendsTab();

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(AppSpace.md),
      children: [
        TextField(
          decoration: InputDecoration(
            hintText: 'Search by name or username',
            prefixIcon: const Icon(Icons.search),
            filled: true,
            fillColor: AppColors.surface,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(AppRadius.md),
              borderSide: BorderSide.none,
            ),
          ),
        ),
        const SectionTitle('Requests'),
        for (final p in requests)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Panel(
              child: Row(children: [
                Avatar(p),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(p.name,
                      style: const TextStyle(fontWeight: FontWeight.w700)),
                ),
                FilledButton(
                  style: FilledButton.styleFrom(
                      minimumSize: const Size(80, 40)),
                  onPressed: () {},
                  child: const Text('Accept'),
                ),
                IconButton(
                    onPressed: () {}, icon: const Icon(Icons.close, size: 20)),
              ]),
            ),
          ),
        SectionTitle('Friends (${friends.length})'),
        for (final p in friends)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Panel(
              child: Row(children: [
                Avatar(p),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(p.name,
                          style: const TextStyle(fontWeight: FontWeight.w700)),
                      Text('${p.streak} day streak  -  form ${p.avgForm.round()}',
                          style: const TextStyle(
                              color: AppColors.textDim, fontSize: 12)),
                    ],
                  ),
                ),
                OutlinedButton(
                  style: OutlinedButton.styleFrom(
                      minimumSize: const Size(72, 38)),
                  onPressed: () {},
                  child: const Text('Challenge'),
                ),
              ]),
            ),
          ),
        const SectionTitle('People you may know'),
        for (final p in suggestions)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Panel(
              child: Row(children: [
                Avatar(p),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(p.name,
                      style: const TextStyle(fontWeight: FontWeight.w700)),
                ),
                OutlinedButton.icon(
                  style: OutlinedButton.styleFrom(
                      minimumSize: const Size(72, 38)),
                  onPressed: () {},
                  icon: const Icon(Icons.person_add_alt, size: 18),
                  label: const Text('Add'),
                ),
              ]),
            ),
          ),
      ],
    );
  }
}

// ---------------- Groups ----------------
class _GroupsTab extends StatelessWidget {
  const _GroupsTab();

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 96),
          children: [
            const Text('My groups',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            for (final g in myGroups) _GroupCard(g),
            const SectionTitle('Discover communities'),
            for (final g in communities) _GroupCard(g),
          ],
        ),
        Positioned(
          right: 16,
          bottom: 16,
          child: FloatingActionButton.extended(
            backgroundColor: AppColors.accent,
            foregroundColor: AppColors.onAccent,
            onPressed: () {},
            icon: const Icon(Icons.add),
            label: const Text('Create group',
                style: TextStyle(fontWeight: FontWeight.w700)),
          ),
        ),
      ],
    );
  }
}

class _GroupCard extends StatelessWidget {
  final Group group;
  const _GroupCard(this.group);

  @override
  Widget build(BuildContext context) {
    final g = group;
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Panel(
        onTap: () => Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => GroupDetailScreen(group: g)),
        ),
        child: Row(children: [
          Container(
            width: 52,
            height: 52,
            decoration: BoxDecoration(
              color: AppColors.surfaceHi,
              borderRadius: BorderRadius.circular(AppRadius.md),
            ),
            child: Icon(g.isCommunity ? Icons.public : Icons.groups_outlined,
                color: AppColors.accent),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Flexible(
                    child: Text(g.name,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                            fontSize: 16, fontWeight: FontWeight.w700)),
                  ),
                  if (g.joined) RoleBadge(g.myRole),
                ]),
                const SizedBox(height: 2),
                Text(
                    '${g.memberCount} members  -  ${g.isPrivate ? 'Private' : 'Public'}',
                    style: const TextStyle(
                        color: AppColors.textDim, fontSize: 13)),
              ],
            ),
          ),
          if (!g.joined)
            OutlinedButton(
              style: OutlinedButton.styleFrom(minimumSize: const Size(64, 38)),
              onPressed: () {},
              child: const Text('Join'),
            ),
        ]),
      ),
    );
  }
}
