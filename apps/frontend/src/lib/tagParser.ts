/**
 * Tag parsing utilities for FloodSafe reports.
 *
 * Reports store tags as a prefix in the description: "[Tag1, Tag2] Description text"
 * These utilities extract and categorize tags for display.
 */

export interface ParsedReport {
    tags: string[];
    description: string;
}

/**
 * Parses a report description to extract tags and clean description.
 *
 * @example
 * parseReportDescription("[Road Blocked, Heavy Rain] Water on main street")
 * // Returns: { tags: ["Road Blocked", "Heavy Rain"], description: "Water on main street" }
 */
export function parseReportDescription(text: string): ParsedReport {
    if (!text) {
        return { tags: [], description: '' };
    }

    const tagMatch = text.match(/^\[([^\]]+)\]\s*/);
    if (tagMatch) {
        const tags = tagMatch[1].split(',').map(t => t.trim()).filter(t => t.length > 0);
        const description = text.slice(tagMatch[0].length);
        return { tags, description };
    }
    return { tags: [], description: text };
}

export type TagCategory = 'emergency' | 'flooding' | 'infrastructure' | 'hazard' | 'default';

// Tag category definitions - must match TAG_CATEGORIES in ReportScreen.tsx
const EMERGENCY_TAGS = ['People Stranded', 'Vehicle Stuck', 'House Flooded', 'Power Outage'];
const FLOODING_TAGS = ['Road Blocked', 'Drainage Overflow', 'Street Flooding', 'Waterlogging', 'Flash Flood', 'Heavy Rain'];
const INFRA_TAGS = ['Bridge Submerged', 'Road Collapse'];
const HAZARD_TAGS = ['Debris Flow', 'Live Wires'];

/**
 * Determines the category of a tag for color coding.
 */
export function getTagCategory(tag: string): TagCategory {
    if (EMERGENCY_TAGS.includes(tag)) return 'emergency';
    if (FLOODING_TAGS.includes(tag)) return 'flooding';
    if (INFRA_TAGS.includes(tag)) return 'infrastructure';
    if (HAZARD_TAGS.includes(tag)) return 'hazard';
    return 'default';
}

/**
 * Returns Tailwind CSS classes for a tag category.
 */
export function getTagCategoryColors(category: TagCategory): string {
    const colors: Record<TagCategory, string> = {
        emergency: 'bg-red-100 text-red-700 border-red-200',
        flooding: 'bg-blue-100 text-blue-700 border-blue-200',
        infrastructure: 'bg-orange-100 text-orange-700 border-orange-200',
        hazard: 'bg-yellow-100 text-yellow-700 border-yellow-200',
        default: 'bg-gray-100 text-gray-700 border-gray-200'
    };
    return colors[category];
}

/**
 * Generates inline HTML for tags (used in MapComponent popup).
 */
export function generateTagHtml(tags: string[]): string {
    if (tags.length === 0) return '';

    return tags.map(tag => {
        const category = getTagCategory(tag);
        // Using inline styles for HTML string context (MapComponent popup)
        const styles: Record<TagCategory, string> = {
            emergency: 'background:#fee2e2;color:#b91c1c;border:1px solid #fecaca;',
            flooding: 'background:#dbeafe;color:#1d4ed8;border:1px solid #bfdbfe;',
            infrastructure: 'background:#ffedd5;color:#c2410c;border:1px solid #fed7aa;',
            hazard: 'background:#fef9c3;color:#a16207;border:1px solid #fef08a;',
            default: 'background:#f3f4f6;color:#374151;border:1px solid #e5e7eb;'
        };
        return `<span style="display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:500;margin-right:4px;margin-bottom:4px;${styles[category]}">${tag}</span>`;
    }).join('');
}
