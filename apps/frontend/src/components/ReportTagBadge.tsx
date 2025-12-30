import { getTagCategory, getTagCategoryColors } from '../lib/tagParser';

interface ReportTagBadgeProps {
    tag: string;
}

/**
 * Displays a report tag as a colored badge.
 * Color is determined by tag category (emergency=red, flooding=blue, etc.)
 */
export function ReportTagBadge({ tag }: ReportTagBadgeProps) {
    const category = getTagCategory(tag);
    const colorClasses = getTagCategoryColors(category);

    return (
        <span className={`px-1.5 py-0.5 rounded text-xs font-medium border ${colorClasses}`}>
            {tag}
        </span>
    );
}

interface ReportTagListProps {
    tags: string[];
}

/**
 * Displays a list of report tags as colored badges.
 */
export function ReportTagList({ tags }: ReportTagListProps) {
    if (tags.length === 0) return null;

    return (
        <div className="flex flex-wrap gap-1 mb-1">
            {tags.map(tag => (
                <ReportTagBadge key={tag} tag={tag} />
            ))}
        </div>
    );
}
