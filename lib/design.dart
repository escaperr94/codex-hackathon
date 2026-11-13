import 'package:material_ui/material_ui.dart';

abstract final class C {
  static const bg = Color(0xFFF6F7F9);
  static const ink = Color(0xFF182028);
  static const muted = Color(0xFF657181);
  static const line = Color(0xFFE2E6ED);
  static const blue = Color(0xFF2454E6);
  static const pale = Color(0xFFEBF0FF);
  static const green = Color(0xFF207660);
  static const amber = Color(0xFF99651C);
  static const red = Color(0xFFAD4B4B);
  static const black = Color(0xFF0D151C);
}

class Label extends StatelessWidget {
  const Label(this.text, {super.key, this.color = C.muted, this.fontSize = 10});
  final String text;
  final Color color;
  final double fontSize;
  @override
  Widget build(BuildContext context) => Text(text.toUpperCase(),
      style: TextStyle(
          color: color,
          fontFamily: 'monospace',
          fontSize: fontSize,
          fontWeight: FontWeight.w800,
          letterSpacing: 1.2));
}

class PageHeading extends StatelessWidget {
  const PageHeading(this.title, {super.key, this.subtitle, this.trailing});
  final String title;
  final String? subtitle;
  final Widget? trailing;
  @override
  Widget build(BuildContext context) =>
      Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Expanded(
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title,
              style: const TextStyle(
                  color: C.ink,
                  fontSize: 31,
                  fontWeight: FontWeight.w800,
                  letterSpacing: -1.4,
                  height: 1.02)),
          if (subtitle != null) ...[
            const SizedBox(height: 8),
            Text(subtitle!,
                style:
                    const TextStyle(color: C.muted, fontSize: 14, height: 1.35))
          ]
        ])),
        if (trailing != null) trailing!
      ]);
}

class Panel extends StatelessWidget {
  const Panel(
      {required this.child,
      super.key,
      this.padding = const EdgeInsets.all(20),
      this.color = Colors.white});
  final Widget child;
  final EdgeInsets padding;
  final Color color;
  @override
  Widget build(BuildContext context) => Container(
      padding: padding,
      decoration: BoxDecoration(
          color: color,
          border: Border.all(color: C.line),
          borderRadius: BorderRadius.circular(22)),
      child: child);
}

class ActionButton extends StatelessWidget {
  const ActionButton(this.label,
      {required this.onPressed,
      super.key,
      this.icon = Icons.arrow_forward_rounded,
      this.secondary = false});
  final String label;
  final VoidCallback? onPressed;
  final IconData icon;
  final bool secondary;
  @override
  Widget build(BuildContext context) => FilledButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: 18),
      label: Text(label),
      style: FilledButton.styleFrom(
          backgroundColor: secondary ? C.pale : C.blue,
          foregroundColor: secondary ? C.blue : Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
          textStyle:
              const TextStyle(fontSize: 13, fontWeight: FontWeight.w800)));
}

class AppPage extends StatelessWidget {
  const AppPage(
      {required this.children,
      super.key,
      this.padding = const EdgeInsets.fromLTRB(22, 14, 22, 20)});
  final List<Widget> children;
  final EdgeInsets padding;
  @override
  Widget build(BuildContext context) => SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      padding: padding,
      child: Column(
          crossAxisAlignment: CrossAxisAlignment.start, children: children));
}

class SectionTitle extends StatelessWidget {
  const SectionTitle(this.title, {super.key, this.trailing});
  final String title;
  final Widget? trailing;
  @override
  Widget build(BuildContext context) => Row(children: [
        Label(title),
        const Spacer(),
        if (trailing != null) trailing!
      ]);
}

class Metric extends StatelessWidget {
  const Metric(this.label, this.value, {super.key, this.accent = C.ink});
  final String label;
  final String value;
  final Color accent;
  @override
  Widget build(BuildContext context) => Expanded(
      child: Container(
          padding: const EdgeInsets.fromLTRB(12, 12, 8, 12),
          decoration: BoxDecoration(
              color: Colors.white,
              border: Border.all(color: C.line),
              borderRadius: BorderRadius.circular(15)),
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Label(label, fontSize: 8),
            const SizedBox(height: 8),
            Text(value,
                style: TextStyle(
                    color: accent,
                    fontFamily: 'monospace',
                    fontSize: 14,
                    fontWeight: FontWeight.w800))
          ])));
}

class StatusDot extends StatelessWidget {
  const StatusDot({super.key, this.color = C.green, this.size = 7});
  final Color color;
  final double size;
  @override
  Widget build(BuildContext context) => Container(
      width: size,
      height: size,
      decoration: BoxDecoration(color: color, shape: BoxShape.circle));
}

class MiniAction extends StatelessWidget {
  const MiniAction({required this.icon, required this.onPressed, super.key});
  final IconData icon;
  final VoidCallback onPressed;
  @override
  Widget build(BuildContext context) => IconButton(
      onPressed: onPressed,
      icon: Icon(icon, size: 20),
      color: C.ink,
      tooltip: 'Action');
}
